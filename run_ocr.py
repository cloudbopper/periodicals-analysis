"""Runs OCR on input PDF's"""

import argparse
import os
import subprocess

def main():
    """Main"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_dir", required=True)
    parser.add_argument("-output_dir", required=True)
    args = parser.parse_args()
    # Find files
    paths = [line[2:] for line in subprocess.check_output("find . -iname '*.pdf'", shell=True).splitlines()]
    # Convert to tiff
    processed_files = set()
    for pdf_filename in paths:
        assert pdf_filename not in processed_files
        processed_files.add(pdf_filename)
        basename = os.path.basename(pdf_filename)
        header = os.path.splitext(basename)[0]
        tiff_filename = "%s/%s.tiff" % (args.output_dir, header)
        cmd = "convert -density 300 '%s' -colorspace gray -depth 8 -strip -background white -alpha off '%s'" % (pdf_filename, tiff_filename)
        print cmd
        subprocess.check_call(cmd, shell=True)
        txt_header = "%s/%s" % (args.output_dir, header)
        cmd = "tesseract '%s' -l eng '%s'" % (tiff_filename, txt_header)
        print cmd
        subprocess.check_call(cmd, shell=True)

if __name__ == "__main__":
    main()
