#!/usr/bin/env python3
"""
receipt_photo.py
----------------
Press a button -> Pi Camera takes a photo -> the photo is tone-mapped and
dithered for a 1-bit thermal head -> printed on an Epson TM-T88V (USB).

Image quality pipeline (the part that actually matters on a 1-bit printer):
  grayscale -> autocontrast -> resize to head width -> gamma tone map
  (dot-gain compensation) -> unsharp pre-sharpen -> dither.

Usage:
  Live mode (waits for button presses):   python3 receipt_photo.py
  Test one capture+print immediately:      python3 receipt_photo.py --test
  Print an existing image (no camera):     python3 receipt_photo.py path/to/img.jpg
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

# The TM-T88V is a 180-dpi head: 512 printable dots (NOT 576 like 203-dpi units).
PRINT_WIDTH = 512

FEED_LINES_AFTER = 3
CUT_PAPER        = True
IMAGE_IMPL       = "bitImageRaster"   # known-good on this printer

# ----------------------------------------------------------------------------
# BUTTON / CAMERA CONFIG
# ----------------------------------------------------------------------------
BUTTON_GPIO   = 17
BUTTON_PULLUP = False   # 3-pin Adeept module drives the pin HIGH on press

ROTATE_DEGREES = 0      # 0/90/180/270 if the camera is mounted rotated

# ----------------------------------------------------------------------------
# IMAGE QUALITY KNOBS  -- tune these to taste
# ----------------------------------------------------------------------------
AUTO_CONTRAST   = True      # normalize tonal range before mapping
CONTRAST_CUTOFF = 2         # % of darkest/lightest pixels to clip

GAMMA = 1.5                 # TONE MAP: >1.0 lightens midtones to counter dot gain.
                            #   Too dark -> raise (1.6-2.0). Too washed out -> lower.

SHARPEN_PERCENT = 150       # unsharp strength; 0 disables. 100-200 is a good range.
SHARPEN_RADIUS  = 2

DITHER = "atkinson"         # "atkinson" (crisp/light), "floyd" (Pillow default), "threshold"

# ----------------------------------------------------------------------------
# TONE MAPPING + DITHERING
# ----------------------------------------------------------------------------

def apply_gamma(gray, gamma):
    """Lift midtones to compensate for thermal dot gain. gamma>1 => lighter."""
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    arr = np.power(arr, 1.0 / gamma)
    return Image.fromarray((arr * 255.0).clip(0, 255).astype(np.uint8), mode="L")


def atkinson_dither(gray):
    """Atkinson error-diffusion: diffuses only 6/8 of the error, so highlights
    stay clean and the print comes out lighter and crisper than Floyd-Steinberg."""
    arr = np.asarray(gray, dtype=np.float32).copy()
    h, w = arr.shape
    neighbours = ((1, 0), (2, 0), (-1, 1), (0, 1), (1, 1), (0, 2))
    for y in range(h):
        for x in range(w):
            old = arr[y, x]
            new = 255.0 if old >= 128 else 0.0
            arr[y, x] = new
            err = (old - new) / 8.0
            for dx, dy in neighbours:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    arr[ny, nx] += err
    out = np.where(arr >= 128, 255, 0).astype(np.uint8)
    return Image.fromarray(out, mode="L").convert("1")


def prepare_image(img):
    img = ImageOps.exif_transpose(img)
    if ROTATE_DEGREES:
        img = img.rotate(ROTATE_DEGREES, expand=True)

    img = img.convert("L")

    if AUTO_CONTRAST:
        img = ImageOps.autocontrast(img, cutoff=CONTRAST_CUTOFF)

    # Resize to the head width first, then map/sharpen at final resolution
    w, h = img.size
    new_h = max(1, round(h * (PRINT_WIDTH / w)))
    img = img.resize((PRINT_WIDTH, new_h), Image.LANCZOS)

    if GAMMA and GAMMA != 1.0:
        img = apply_gamma(img, GAMMA)

    if SHARPEN_PERCENT:
        img = img.filter(ImageFilter.UnsharpMask(
            radius=SHARPEN_RADIUS, percent=SHARPEN_PERCENT, threshold=2))

    if DITHER == "atkinson":
        img = atkinson_dither(img)
    elif DITHER == "floyd":
        img = img.convert("1")
    else:  # plain threshold
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
        time.sleep(1.0)  # let exposure / white balance settle

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
