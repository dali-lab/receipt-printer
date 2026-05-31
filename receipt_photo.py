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
  MAX7219 8x8 LED matrix over SPI. Lights before capture, gives the camera
  a beat to adjust exposure, then turns off (flash sync). Degrades gracefully
  to no-flash if the matrix isn't connected / libraries aren't installed.

Setup (one time):
  sudo raspi-config nonint do_spi 0 && sudo reboot
  uv pip install luma.led_matrix spidev rpi-lgpio

Usage:
  Live (waits for button):              python3 receipt_photo.py
  Test one capture+print immediately:   python3 receipt_photo.py --test
  Print an existing image (no camera):  python3 receipt_photo.py path/to/img.jpg

Wiring (MAX7219 matrix -> Pi, hardware SPI):
  VCC->5V(pin 2)  GND->GND(pin 6)  DIN->GPIO10/MOSI(pin 19)
  CS->GPIO8/CE0(pin 24)  CLK->GPIO11/SCLK(pin 23)
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
BUTTON_PULLUP = False
ROTATE_DEGREES = 0
CAMERA_SETTLE  = 1.5       # seconds to settle when the camera first starts

# ----------------------------------------------------------------------------
# FLASH CONFIG (MAX7219 8x8 LED matrix over SPI)
# ----------------------------------------------------------------------------
FLASH_ENABLED   = True
FLASH_SPI_PORT  = 0        # /dev/spidev0.0  -> port 0, device 0
FLASH_SPI_DEV   = 0
FLASH_CASCADED  = 1        # number of chained 8x8 panels
FLASH_BRIGHTNESS = 255     # 0-255 (contrast)
FLASH_SETTLE    = 0.4      # seconds the light is on before capture, so
                           # auto-exposure adapts to the lit scene

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
FLAT_EDGE_THRESHOLD  = 12  # Laplacian threshold to classify a pixel as an edge (0-255)
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
# FLASH (MAX7219 LED matrix)
# ----------------------------------------------------------------------------

class Flash:
    """LED-matrix flash. If anything is missing/unconnected, on()/off() become
    no-ops so the rest of the program keeps working."""

    def __init__(self):
        self.device = None
        if not FLASH_ENABLED:
            return
        try:
            from luma.led_matrix.device import max7219
            from luma.core.interface.serial import spi, noop
            serial = spi(port=FLASH_SPI_PORT, device=FLASH_SPI_DEV, gpio=noop())
            self.device = max7219(serial, cascaded=FLASH_CASCADED)
            self.device.contrast(FLASH_BRIGHTNESS)
            self.device.clear()
        except Exception as e:
            print(f"Flash disabled (matrix not available): {e}")
            self.device = None

    def on(self):
        if self.device is None:
            return
        try:
            from luma.core.render import canvas
            with canvas(self.device) as draw:
                draw.rectangle(self.device.bounding_box, fill="white")
        except Exception:
            pass

    def off(self):
        if self.device is None:
            return
        try:
            self.device.clear()
        except Exception:
            pass

    def close(self):
        self.off()


# ----------------------------------------------------------------------------
# PRINTER
# ----------------------------------------------------------------------------

def get_printer():
    from escpos.printer import Usb
    kwargs = {"profile": PRINTER_PROFILE} if PRINTER_PROFILE else {}
    return Usb(PRINTER_VENDOR_ID, PRINTER_PRODUCT_ID, **kwargs)


def print_image(img):
    printer = get_printer()
    try:
        printer.image(img, impl=IMAGE_IMPL)
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
    button = Button(BUTTON_GPIO, pull_up=BUTTON_PULLUP)
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