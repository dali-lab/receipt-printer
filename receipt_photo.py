#!/usr/bin/env python3
"""
receipt_photo.py
----------------
Press a button -> Pi Camera takes a photo -> the photo is tone-mapped and
dithered for a 1-bit thermal head -> printed on an Epson TM-T88V (USB).

Quality pipeline:
  grayscale -> resize -> CLAHE (adaptive local equalization) ->
  (auto)gamma tone map -> clarity -> unsharp edges -> serpentine Atkinson dither.

Usage:
  Live (waits for button):              python3 receipt_photo.py
  Test one capture+print immediately:   python3 receipt_photo.py --test
  Print an existing image (no camera):  python3 receipt_photo.py path/to/img.jpg
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
CAMERA_SETTLE  = 1.5

# ----------------------------------------------------------------------------
# IMAGE QUALITY KNOBS
# ----------------------------------------------------------------------------
# CLAHE: adaptive local contrast (the main detail step)
CLAHE_CLIP  = 2.0          # higher = stronger local contrast (2-4 typical); too high = noisy
CLAHE_TILES = 8            # grid is TILES x TILES; more tiles = more local
CONTRAST_CUTOFF = 2        # only used by the fallback if OpenCV is missing

# Tone mapping (dot-gain compensation)
AUTO_GAMMA   = True
GAMMA_TARGET = 0.60        # target mean brightness (0..1); higher = lighter print
GAMMA        = 1.5         # used only when AUTO_GAMMA = False

# Clarity (mid-frequency detail; CLAHE already adds local contrast, so keep modest)
LOCAL_CONTRAST_AMOUNT = 0.35
LOCAL_CONTRAST_RADIUS = 12

# Edge sharpening
SHARPEN_PERCENT = 150
SHARPEN_RADIUS  = 2

DITHER = "atkinson"        # "atkinson" | "floyd" | "threshold"

# ----------------------------------------------------------------------------
# CONTRAST / TONE / DITHER
# ----------------------------------------------------------------------------

def clahe(gray):
    """Contrast-limited adaptive histogram equalization (per-tile local contrast).
    Falls back to global autocontrast if OpenCV isn't installed."""
    try:
        import cv2
    except ImportError:
        return ImageOps.autocontrast(gray, cutoff=CONTRAST_CUTOFF)
    arr = np.asarray(gray, np.uint8)
    op = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_TILES, CLAHE_TILES))
    return Image.fromarray(op.apply(arr), "L")


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

    img = clahe(img)

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

def handle_press(camera):
    if not _busy.acquire(blocking=False):
        print("Still printing the last one -- ignoring press.")
        return
    try:
        print("Capturing...")
        raw = camera.capture()
        img = prepare_image(raw)
        print(f"Printing {img.width}x{img.height} ...")
        print_image(img)
        print("Done.")
    except Exception as e:
        print(f"Error during capture/print: {e}")
    finally:
        _busy.release()


def run_live():
    from gpiozero import Button
    from signal import pause

    camera = Camera()
    button = Button(BUTTON_GPIO, pull_up=BUTTON_PULLUP)
    button.when_pressed = lambda: handle_press(camera)

    print(f"Ready. Press the button on GPIO{BUTTON_GPIO} to take + print a photo.")
    print("Ctrl+C to quit.")
    try:
        pause()
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
        print("\nExiting.")


def main():
    args = sys.argv[1:]
    if args and args[0] != "--test":
        print(f"Preparing and printing {args[0]} ...")
        print_image(prepare_image(Image.open(args[0])))
        print("Done.")
        return
    if args and args[0] == "--test":
        camera = Camera()
        try:
            handle_press(camera)
        finally:
            camera.close()
        return
    run_live()


if __name__ == "__main__":
    main()
