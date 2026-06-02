#!/usr/bin/env python3
"""
receipt_photo.py
----------------
Press a button -> LED-matrix flash fires -> Pi Camera takes a photo ->
the photo is tone-mapped and dithered for a 1-bit thermal head ->
printed on an Epson TM-T88V (USB).

Quality pipeline:
  grayscale -> resize -> CLAHE (adaptive local equalization) ->
  (auto)gamma tone map -> clarity -> unsharp edges -> serpentine Atkinson dither.

Flash:
  Adeept common-anode RGB LED module (pins: + R G B). All three cathodes are
  pulled LOW simultaneously to mix to white, then HIGH to turn off. Built-in
  series resistors keep current safe (~14 mA per color, ~40 mA total when
  white) -- no current-budget concerns. Degrades gracefully to no-flash if
  gpiozero isn't available.

  NOTE: this is a single 5mm RGB LED, not a matrix. It is much dimmer than a
  real flash and contributes little to photo exposure -- but the light is
  white (correct color temperature) rather than red.

Setup (one time):
  uv pip install rpi-lgpio

Usage:
  Live (waits for button):              python3 receipt_photo.py
  Test one capture+print immediately:   python3 receipt_photo.py --test
  Print an existing image (no camera):  python3 receipt_photo.py path/to/img.jpg

Wiring (Adeept RGB LED module -> Pi):
  +  -> 3.3V  (pin 1)    -- common anode. 3.3V (not 5V) so the "off" state
                            (GPIO HIGH = 3.3V on cathode) gives 0V across
                            the LED -> no idle glow. With 5V here, red's
                            low forward voltage leaks current and the LED
                            idles dim red.
  R  -> GPIO22 (pin 15)
  G  -> GPIO23 (pin 16)
  B  -> GPIO24 (pin 18)

  Tradeoff vs. 5V supply: peak brightness is lower and color skews warm
  (red has the lowest Vf so it gets the most current at 3.3V). Acceptable
  for a tiny indicator flash; for real illumination, switch to a MAX7219
  white matrix.
"""

import sys
import time
import threading

import numpy as np
from PIL import Image, ImageOps, ImageFilter

# ----------------------------------------------------------------------------
# PRINTER CONFIG  (confirmed for this TM-T88V)
# ----------------------------------------------------------------------------
PRINTER_VENDOR_ID  = 0x04b8
PRINTER_PRODUCT_ID = 0x0202
PRINTER_PROFILE    = "TM-T88V"

PRINT_WIDTH = 512          # 180-dpi head = 512 printable dots
FEED_LINES_AFTER = 3
CUT_PAPER        = True
IMAGE_IMPL       = "bitImageRaster"

# ----------------------------------------------------------------------------
# BUTTON / CAMERA CONFIG
# ----------------------------------------------------------------------------
BUTTON_GPIO   = 17
BUTTON_PULLUP = True       # internal pull-up; wire button between GPIO17 and GND
ROTATE_DEGREES = 0
CAMERA_SETTLE  = 1.5       # seconds to settle when the camera first starts

# ----------------------------------------------------------------------------
# FLASH CONFIG (Adeept common-anode RGB LED module)
# ----------------------------------------------------------------------------
FLASH_ENABLED    = True
FLASH_RED_GPIO   = 22      # BCM numbering; physical pin 15
FLASH_GREEN_GPIO = 23      # BCM numbering; physical pin 16
FLASH_BLUE_GPIO  = 24      # BCM numbering; physical pin 18
FLASH_SETTLE     = 0.4     # seconds the light is on before capture, so
                           # auto-exposure adapts to the lit scene

# Countdown sequence played before each capture. Each tuple is
# (red, green, blue, on_seconds, off_seconds). Default: red -> yellow ->
# green -> three rapid white blinks -> solid white for capture.
COUNTDOWN_STEPS = (
    (1, 0, 0, 0.25, 0.75),   # 3 ... red
    (1, 1, 0, 0.25, 0.75),   # 2 ... yellow
    (0, 1, 0, 0.25, 0.75),   # 1 ... green
    (1, 1, 1, 0.08, 0.08),   # go!
    (1, 1, 1, 0.08, 0.08),
    (1, 1, 1, 0.08, 0.08),
)

# ----------------------------------------------------------------------------
# IMAGE QUALITY KNOBS
# ----------------------------------------------------------------------------
CLAHE_CLIP  = 2.0          # local contrast strength (2-4); too high = noisy
CLAHE_TILES = 8
CONTRAST_CUTOFF = 2        # fallback only, if OpenCV missing

AUTO_GAMMA   = True
GAMMA_TARGET = 0.60        # target mean brightness (0..1); higher = lighter
GAMMA        = 1.5         # used only when AUTO_GAMMA = False

LOCAL_CONTRAST_AMOUNT = 0.35
LOCAL_CONTRAST_RADIUS = 12

SHARPEN_PERCENT = 150
SHARPEN_RADIUS  = 2

DENOISE_RADIUS  = 2        # bilateral spatial sigma; keep small (1-3) for light touch
DENOISE_SIGMA   = 10       # bilateral intensity sigma; keep low (8-15) to preserve edges

FLAT_DENOISE         = True
FLAT_PATCH_SIZE      = 5   # patch radius in pixels; smaller = more precise, less smoothing
                           # try 3 (precise) to 7 (smoother flat surfaces)
FLAT_EDGE_THRESHOLD  = 30  # Laplacian threshold to classify a pixel as an edge (0-255)
                           # lower = more pixels treated as edges (more detail preserved)
                           # higher = fewer edges detected (more area gets smoothed)

DITHER = "atkinson"        # "atkinson" | "floyd" | "threshold"

# ----------------------------------------------------------------------------
# CONTRAST / TONE / DITHER
# ----------------------------------------------------------------------------

def clahe(gray):
    """Adaptive local contrast. Falls back to global autocontrast w/o OpenCV."""
    try:
        import cv2
    except ImportError:
        return ImageOps.autocontrast(gray, cutoff=CONTRAST_CUTOFF)
    arr = np.asarray(gray, np.uint8)
    op = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_TILES, CLAHE_TILES))
    return Image.fromarray(op.apply(arr), "L")


def denoise(gray):
    """Light bilateral filter pass. Falls back to median if OpenCV is missing."""
    try:
        import cv2
        arr = np.asarray(gray, np.uint8)
        d = DENOISE_RADIUS * 2 + 1
        out = cv2.bilateralFilter(arr, d, DENOISE_SIGMA, DENOISE_SIGMA)
        return Image.fromarray(out, "L")
    except ImportError:
        return gray.filter(ImageFilter.MedianFilter(size=3))


def flat_area_denoise(gray):
    """Patch-safe flat denoiser: averages a pixel only if its entire surrounding
    patch contains no edges. If any edge falls inside the patch, the center pixel
    is left untouched. This prevents edge blur regardless of patch size."""
    try:
        import cv2
    except ImportError:
        return gray

    arr = np.asarray(gray, np.float32)

    # Detect edges across the whole image.
    lap = cv2.Laplacian(arr.astype(np.uint8), cv2.CV_32F)
    edge_map = (np.abs(lap) > FLAT_EDGE_THRESHOLD).astype(np.float32)

    r = FLAT_PATCH_SIZE
    kernel = np.ones((2 * r + 1, 2 * r + 1), np.float32)

    # Sum of edge pixels within each pixel's patch window.
    # If this sum is 0, the patch is entirely edge-free.
    edge_sum = cv2.filter2D(edge_map, -1, kernel, borderType=cv2.BORDER_REFLECT)
    flat_patch = (edge_sum == 0)

    # Box-average of pixel values within the patch.
    patch_sum   = cv2.filter2D(arr, -1, kernel, borderType=cv2.BORDER_REFLECT)
    patch_count = (2 * r + 1) ** 2
    smoothed    = patch_sum / patch_count

    # Only replace pixels whose entire patch was edge-free.
    flat_count = int(flat_patch.sum())
    total = arr.size
    print(f"flat_area_denoise: {flat_count}/{total} pixels smoothed ({100*flat_count/total:.1f}%) "
          f"patch={2*r+1}x{2*r+1} edge_threshold={FLAT_EDGE_THRESHOLD}")

    out = np.where(flat_patch, smoothed, arr).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out, "L")


def pick_gamma(gray):
    mean = float(np.clip(np.asarray(gray, np.float32).mean() / 255.0, 1e-3, 0.999))
    g = np.log(mean) / np.log(GAMMA_TARGET)
    return float(np.clip(g, 0.5, 3.0))


def apply_gamma(gray, gamma):
    arr = np.asarray(gray, np.float32) / 255.0
    arr = np.power(arr, 1.0 / gamma)
    return Image.fromarray((arr * 255.0).clip(0, 255).astype(np.uint8), "L")


def local_contrast(gray, radius, amount):
    if amount <= 0:
        return gray
    blurred = gray.filter(ImageFilter.GaussianBlur(radius))
    a = np.asarray(gray, np.float32)
    b = np.asarray(blurred, np.float32)
    out = a + amount * (a - b)
    return Image.fromarray(out.clip(0, 255).astype(np.uint8), "L")


def atkinson_dither(gray):
    """Serpentine Atkinson error diffusion."""
    arr = np.asarray(gray, np.float32).copy()
    h, w = arr.shape
    base = ((1, 0), (2, 0), (-1, 1), (0, 1), (1, 1), (0, 2))
    for y in range(h):
        ltr = (y % 2 == 0)
        xs = range(w) if ltr else range(w - 1, -1, -1)
        for x in xs:
            old = arr[y, x]
            new = 255.0 if old >= 128 else 0.0
            arr[y, x] = new
            err = (old - new) / 8.0
            for dx, dy in base:
                ddx = dx if ltr else -dx
                nx, ny = x + ddx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    arr[ny, nx] += err
    out = np.where(arr >= 128, 255, 0).astype(np.uint8)
    return Image.fromarray(out, "L").convert("1")


def prepare_image(img):
    img = ImageOps.exif_transpose(img)
    if ROTATE_DEGREES:
        img = img.rotate(ROTATE_DEGREES, expand=True)

    img = img.convert("L")

    w, h = img.size
    new_h = max(1, round(h * (PRINT_WIDTH / w)))
    img = img.resize((PRINT_WIDTH, new_h), Image.LANCZOS)

    img = denoise(img)
    img = clahe(img)
    img = denoise(img)
    if FLAT_DENOISE:
        img = flat_area_denoise(img)

    gamma = pick_gamma(img) if AUTO_GAMMA else GAMMA
    if gamma and gamma != 1.0:
        img = apply_gamma(img, gamma)

    img = local_contrast(img, LOCAL_CONTRAST_RADIUS, LOCAL_CONTRAST_AMOUNT)

    if SHARPEN_PERCENT:
        img = img.filter(ImageFilter.UnsharpMask(
            radius=SHARPEN_RADIUS, percent=SHARPEN_PERCENT, threshold=2))

    if DITHER == "atkinson":
        img = atkinson_dither(img)
    elif DITHER == "floyd":
        img = img.convert("1")
    else:
        img = img.point(lambda p: 255 if p >= 128 else 0).convert("1")

    return img


# ----------------------------------------------------------------------------
# FLASH (Adeept common-anode RGB LED)
# ----------------------------------------------------------------------------

class Flash:
    """Adeept common-anode RGB LED used as a small white flash. Each cathode
    is driven by a GPIO with inverted polarity (active_high=False): LED.on()
    pulls the cathode LOW so current flows; LED.off() drives it HIGH (3.3 V)
    which sits below the LED forward voltage with 5 V on the anode, so the
    LED is fully off with no leakage. Degrades to no-op if gpiozero isn't
    available."""

    def __init__(self):
        self.pins = ()
        if not FLASH_ENABLED:
            return
        try:
            from gpiozero import LED
            self.pins = (
                LED(FLASH_RED_GPIO,   active_high=False, initial_value=False),
                LED(FLASH_GREEN_GPIO, active_high=False, initial_value=False),
                LED(FLASH_BLUE_GPIO,  active_high=False, initial_value=False),
            )
        except Exception as e:
            print(f"Flash disabled (gpiozero not available): {e}")
            self.pins = ()

    def _set(self, r, g, b):
        if not self.pins:
            return
        for pin, val in zip(self.pins, (r, g, b)):
            try:
                pin.on() if val else pin.off()
            except Exception:
                pass

    def on(self):
        self._set(1, 1, 1)

    def off(self):
        self._set(0, 0, 0)

    def countdown(self):
        """Play COUNTDOWN_STEPS as a visual cue, then leave the LED off."""
        if not self.pins:
            return
        for r, g, b, on_s, off_s in COUNTDOWN_STEPS:
            self._set(r, g, b)
            time.sleep(on_s)
            self.off()
            time.sleep(off_s)

    def close(self):
        self.off()
        for pin in self.pins:
            try:
                pin.close()
            except Exception:
                pass
        self.pins = ()


# ----------------------------------------------------------------------------
# CAPTION
# ----------------------------------------------------------------------------

CAPTION_TEXT    = "Technigala 26S"
CAPTION_FONT_SIZE = 24          # px; Press Start 2P is 8-px grid so multiples of 8 look sharpest
CAPTION_PADDING = 18            # px of white space above and below the text row
CAPTION_FONT_URL = (
    "https://github.com/google/fonts/raw/main/ofl/pressstart2p/PressStart2P-Regular.ttf"
)

_caption_font = None

def _load_pixel_font():
    """Return an ImageFont for the caption. Downloads Press Start 2P once to /tmp."""
    global _caption_font
    if _caption_font is not None:
        return _caption_font

    from PIL import ImageFont
    import os, urllib.request

    cache = "/tmp/PressStart2P-Regular.ttf"
    if not os.path.exists(cache):
        print("Downloading pixel font...")
        urllib.request.urlretrieve(CAPTION_FONT_URL, cache)

    try:
        _caption_font = ImageFont.truetype(cache, CAPTION_FONT_SIZE)
    except Exception as e:
        print(f"Could not load pixel font ({e}); falling back to default.")
        _caption_font = ImageFont.load_default()
    return _caption_font


def make_caption_strip(width):
    """Return a 1-bit PIL image of CAPTION_TEXT centred at the given width."""
    from PIL import ImageDraw
    font = _load_pixel_font()

    # Measure text
    probe = Image.new("L", (1, 1))
    draw  = ImageDraw.Draw(probe)
    bbox  = draw.textbbox((0, 0), CAPTION_TEXT, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    strip_h = int(th + CAPTION_PADDING * 2)
    strip   = Image.new("L", (width, strip_h), 255)   # white background
    draw    = ImageDraw.Draw(strip)
    x = (width - tw) // 2
    y = CAPTION_PADDING
    draw.text((x, y), CAPTION_TEXT, font=font, fill=0)

    # Threshold to hard 1-bit so it matches the dithered photo
    arr = np.asarray(strip, np.uint8)
    arr = np.where(arr < 128, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "L").convert("1")


# ----------------------------------------------------------------------------
# PRINTER
# ----------------------------------------------------------------------------

def get_printer():
    from escpos.printer import Usb
    kwargs = {"profile": PRINTER_PROFILE} if PRINTER_PROFILE else {}
    return Usb(PRINTER_VENDOR_ID, PRINTER_PRODUCT_ID, **kwargs)


def print_image(img):
    caption = make_caption_strip(img.width)

    # Stack photo above caption
    combined = Image.new("1", (img.width, img.height + caption.height), 1)
    combined.paste(img,     (0, 0))
    combined.paste(caption, (0, img.height))

    printer = get_printer()
    try:
        printer.image(combined, impl=IMAGE_IMPL)
        if FEED_LINES_AFTER:
            printer.print_and_feed(FEED_LINES_AFTER)
        if CUT_PAPER:
            printer.cut()
    finally:
        try:
            printer.close()
        except Exception:
            pass


# ----------------------------------------------------------------------------
# CAMERA
# ----------------------------------------------------------------------------

class Camera:
    def __init__(self):
        from picamera2 import Picamera2
        self.cam = Picamera2()
        self.cam.configure(self.cam.create_still_configuration())
        self.cam.start()
        time.sleep(CAMERA_SETTLE)

    def capture(self):
        return self.cam.capture_image("main")

    def close(self):
        try:
            self.cam.stop()
        except Exception:
            pass


# ----------------------------------------------------------------------------
# MAIN FLOW
# ----------------------------------------------------------------------------

_busy = threading.Lock()

def handle_press(camera, flash):
    if not _busy.acquire(blocking=False):
        print("Still printing the last one -- ignoring press.")
        return
    try:
        print("Countdown...")
        flash.countdown()
        print("Flash on, capturing...")
        flash.on()
        time.sleep(FLASH_SETTLE)        # let auto-exposure adapt to the light
        try:
            raw = camera.capture()
        finally:
            flash.off()
        img = prepare_image(raw)
        print(f"Printing {img.width}x{img.height} ...")
        print_image(img)
        print("Done.")
    except Exception as e:
        print(f"Error during capture/print: {e}")
        flash.off()
    finally:
        _busy.release()


def run_live():
    from gpiozero import Button
    from signal import pause

    camera = Camera()
    flash = Flash()
    button = Button(BUTTON_GPIO, pull_up=BUTTON_PULLUP, bounce_time=0.05)
    button.when_pressed = lambda: handle_press(camera, flash)

    print(f"Ready. Press the button on GPIO{BUTTON_GPIO} to take + print a photo.")
    print("Ctrl+C to quit.")
    try:
        pause()
    except KeyboardInterrupt:
        pass
    finally:
        flash.close()
        camera.close()
        print("\nExiting.")


def main():
    args = sys.argv[1:]

    if args and args[0] != "--test":
        # Print an existing image file (no camera, no flash)
        print(f"Preparing and printing {args[0]} ...")
        print_image(prepare_image(Image.open(args[0])))
        print("Done.")
        return

    if args and args[0] == "--test":
        camera = Camera()
        flash = Flash()
        try:
            handle_press(camera, flash)
        finally:
            flash.close()
            camera.close()
        return

    run_live()


if __name__ == "__main__":
    main()