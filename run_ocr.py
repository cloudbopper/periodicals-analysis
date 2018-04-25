"""Runs OCR on input PDF's"""

import argparse
import os
import subprocess

def main():
    """Main"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_dir", required=True)
    parser.add_argument("-output_dir", required=True)
    parser.add_argument("-image_type", default="tiff", choices=["tiff", "png"])
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.image_type == "png":
        raise NotImplementedError("PNG support requires handling multiple image file per input PDF")
    # Find files
    paths = [line[2:] for line in subprocess.check_output("find . -iname '*.pdf'", shell=True).splitlines()]
    print "Number of input PDFs: %d" % len(paths)
    # Process files
    processed_files = set()
    for idx, pdf_filename in enumerate(paths):
        print "Processing PDF %d of %d: %s" % (idx + 1, len(paths), pdf_filename)
        assert pdf_filename not in processed_files
        processed_files.add(pdf_filename)
        basename = os.path.basename(pdf_filename)
        header = os.path.splitext(basename)[0]
        image_filename = "%s/%s.%s" % (args.output_dir, header, args.image_type)
        cmd = "convert -density 300 '%s' -colorspace gray -depth 8 -strip -background white -alpha off '%s'" % (pdf_filename, image_filename)
        print cmd
        subprocess.check_call(cmd, shell=True)
        txt_header = "%s/%s" % (args.output_dir, header)
        cmd = "tesseract '%s' -l eng '%s'" % (image_filename, txt_header)
        print cmd
        subprocess.check_call(cmd, shell=True)

if __name__ == "__main__":
    main()
