#!/usr/bin/env python3
"""
receipt_photo.py
----------------
Press a button -> Pi Camera takes a photo -> photo is converted to a
1-bit dithered image at the printer's working width -> printed on an
80mm USB ESC/POS thermal receipt printer.

Hardware assumptions:
  * Raspberry Pi Camera Module (CSI ribbon) -> uses picamera2
  * 80mm thermal printer over USB, ESC/POS compatible -> uses python-escpos
  * Momentary push button wired between a GPIO pin and GND (gpiozero
    default is internal pull-up, so no extra resistor needed)

Usage:
  Live mode (waits for button presses):   python3 receipt_photo.py
  Test one capture+print immediately:      python3 receipt_photo.py --test
  Print an existing image (no camera):     python3 receipt_photo.py path/to/img.jpg
"""

import sys
import time
import threading

from PIL import Image, ImageOps

# ----------------------------------------------------------------------------
# CONFIG  -- edit these to match your setup
# ----------------------------------------------------------------------------

# USB printer vendor/product IDs. Find yours by running `lsusb` in a terminal:
#   e.g. "Bus 001 Device 005: ID 0416:5011 ..."  ->  VENDOR=0x0416 PRODUCT=0x5011
PRINTER_VENDOR_ID  = 0x0416
PRINTER_PRODUCT_ID = 0x5011
# Some printers also need a profile, e.g. PRINTER_PROFILE = "TM-T88V".
PRINTER_PROFILE    = None

# GPIO pin (BCM numbering) the button is connected to. GPIO17 = physical pin 11.
BUTTON_GPIO = 17

# Printable width in dots. 80mm heads almost always have a 72mm printable
# area at 203 dpi = 576 dots. (58mm printers would be 384.)
PRINT_WIDTH = 576

# Image tuning ---------------------------------------------------------------
ROTATE_DEGREES   = 0      # rotate the photo if the camera is mounted sideways (0/90/180/270)
AUTO_CONTRAST    = True   # stretch contrast -- helps a lot on thermal paper
CONTRAST_CUTOFF  = 2      # percent of darkest/lightest pixels to clip for autocontrast
FEED_LINES_AFTER = 3      # blank lines fed after the image before the cut
CUT_PAPER        = True   # set False if your printer has no auto-cutter

# ----------------------------------------------------------------------------
# IMAGE PROCESSING
# ----------------------------------------------------------------------------

def prepare_image(img: Image.Image) -> Image.Image:
    """Turn any input image into a 576px-wide, 1-bit dithered image ready to print."""
    # Respect EXIF orientation (phones/cameras often store rotation as metadata)
    img = ImageOps.exif_transpose(img)

    if ROTATE_DEGREES:
        img = img.rotate(ROTATE_DEGREES, expand=True)

    # To grayscale
    img = img.convert("L")

    if AUTO_CONTRAST:
        img = ImageOps.autocontrast(img, cutoff=CONTRAST_CUTOFF)

    # Resize to the printer's width, keeping aspect ratio
    w, h = img.size
    new_h = max(1, round(h * (PRINT_WIDTH / w)))
    img = img.resize((PRINT_WIDTH, new_h), Image.LANCZOS)

    # Floyd-Steinberg dither down to 1-bit (this is what gives the "grayscale" look)
    img = img.convert("1")
    return img


# ----------------------------------------------------------------------------
# PRINTER
# ----------------------------------------------------------------------------

def get_printer():
    """Open the USB ESC/POS printer."""
    from escpos.printer import Usb
    kwargs = {}
    if PRINTER_PROFILE:
        kwargs["profile"] = PRINTER_PROFILE
    return Usb(PRINTER_VENDOR_ID, PRINTER_PRODUCT_ID, **kwargs)


def print_image(img: Image.Image):
    """Send a prepared PIL image to the printer."""
    printer = get_printer()
    try:
        # bitImageRaster is the most widely compatible raster mode
        printer.image(img, impl="bitImageRaster")
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
    """Thin wrapper around picamera2 that stays warm between shots."""

    def __init__(self):
        from picamera2 import Picamera2
        self.cam = Picamera2()
        # A still config gives you the full sensor resolution.
        config = self.cam.create_still_configuration()
        self.cam.configure(config)
        self.cam.start()
        time.sleep(1.0)  # let auto-exposure / white balance settle

    def capture(self) -> Image.Image:
        # capture_image returns a PIL Image directly (no temp file needed)
        return self.cam.capture_image("main")

    def close(self):
        try:
            self.cam.stop()
        except Exception:
            pass


# ----------------------------------------------------------------------------
# MAIN FLOW
# ----------------------------------------------------------------------------

# Guard so a second press while we're still printing is ignored
_busy = threading.Lock()

def handle_press(camera: Camera):
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
    button = Button(BUTTON_GPIO)  # default: pull-up, wire button to GND
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

    # Print an existing image file, no camera involved
    if args and args[0] not in ("--test",):
        path = args[0]
        print(f"Preparing and printing {path} ...")
        print_image(prepare_image(Image.open(path)))
        print("Done.")
        return

    # One-shot capture+print, useful for testing the camera+printer chain
    if args and args[0] == "--test":
        camera = Camera()
        try:
            handle_press(camera)
        finally:
            camera.close()
        return

    # Normal mode: wait for button presses
    run_live()


if __name__ == "__main__":
    main()
