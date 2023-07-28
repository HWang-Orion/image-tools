import cv2 as cv
import argparse
import os
from colorama import Fore, Style
from tqdm import tqdm

import utils
from functions import *

THRESHOLD = 8  # threshold of the maximum image folder size in GB (for program automatic mode check only)


def main(args):
    steps_finished = 0
    steps_to_do = int(args.add_watermark) + int(args.resize) + int(args.padding)
    base_path = os.getcwd()
    img_path = os.path.join(base_path, args.image_path)
    assert os.path.exists(img_path), ValueError("Wrong image path given: " + img_path)

    # check num of images first
    files = os.listdir(img_path)
    image_files = list(filter(lambda x: x.endswith('.jpg'), files))
    print(Fore.GREEN + f"{len(image_files)} image(s) founded in {img_path}.")
    print(Style.RESET_ALL)
    
    run_mode = args.run_mode
    assert run_mode in ("together", "one_by_one", "check_first")
    if run_mode == "check_first":
        if utils.check_folder_size(img_path) >= THRESHOLD:
            run_mode = "one_by_one"
        else:
            run_mode = "together"
    if run_mode == "together":
        images = utils.read_images(img_path, image_files)
    
    assert 1 <= args.output_quality <= 100, ValueError(f"Output quality must be in $[1, 100]$, but got {args.output_quality} instead.")
    
    out_path = os.path.join(base_path, args.output_path)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    
    if args.add_watermark:
        print("\n--------------------------------")
        print("        Adding watermark        ")
        print("--------------------------------\n")
        steps_finished += 1
        wm_path = os.path.join(os.getcwd(), args.wm_folder)
        if args.watermark_mode == "preset_watermarks":
            assert os.path.exists(wm_path), ValueError("Watermark path not found!")
            if not args.no_apparent:
                try:
                    wm_app = cv.imread(os.path.join(wm_path, args.apparent_wm_filename), cv.IMREAD_UNCHANGED)
                except FileNotFoundError:
                    raise ValueError("Apparent watermark not found!")
                wm_app_pos = args.apparent_wm_position
                wm_app_ori = args.apparent_wm_orientation
                wm_app_scl = args.apparent_wm_scale
            if not args.no_dark:
                try:
                    wm_dark = cv.imread(os.path.join(wm_path, args.dark_wm_filename), cv.IMREAD_UNCHANGED)
                except FileNotFoundError:
                    raise ValueError("Dark watermark not found!")
                wm_dark_scl = args.dark_wm_scale
            if run_mode == "one_by_one":
                print(Fore.YELLOW + "Please do not modify the output folder including its contents: " + out_path)
                print(Style.RESET_ALL)
                for img in tqdm(os.listdir(img_path)):
                    im = cv.imread(os.path.join(img_path, img), cv.IMREAD_UNCHANGED)
                    if not args.no_apparent:
                        im = wm_preset.add_watermark(wm_app, im,
                                                     wm_pos=wm_app_pos, wm_scale=wm_app_scl, wm_ori=wm_app_ori)
                    if not args.no_dark:
                        im = wm_preset.add_watermark(wm_dark, im, wm_scale=wm_dark_scl)

                    quality = 100 if steps_finished != steps_to_do else args.output_quality
                    cv.imwrite(os.path.join(out_path, img), im, [int(cv.IMWRITE_JPEG_QUALITY), quality])
                img_path = out_path
            else:
                if not args.no_apparent:
                    images = wm_preset.add_watermark(wm_app, images,
                                                     wm_pos=wm_app_pos, wm_scale=wm_app_scl, wm_ori=wm_app_ori)
                if not args.no_dark:
                    images = wm_preset.add_watermark(wm_dark, images, wm_scale=wm_dark_scl)
        elif args.watermark_mode == "text":
            assert len(args.dark_wm_text) > 0, ValueError("Watermark text invalid!")
            # todo add text watermarks
            raise NotImplementedError("This approach is not implemented yet.")
        else:
            raise ValueError("Wrong watermark mode given!")
    
    if args.resize:
        print("\n--------------------------------")
        print("            Resizing            ")
        print("--------------------------------\n")
        steps_finished += 1
        if args.resize_resolution in args.resize_resolution:
            resolutions = {
                "2k": (1080, 1920),
                "2.5k": (1440, 2560),
                "4k": (2160, 3840),
                "c4k": (2160, 4096),
                "8k": (4320, 7680)
            }
            resol = resolutions[args.resize_resolution]
        elif 'x' in args.resize_resolution:
            resol = list(args.resize_resolution.split('x', 1))
            if not isinstance(resol[0], int) or not isinstance(resol[1], float):
                raise ValueError("Invalid resize resolution! Expected to be int or preset keywords but got" + args.resize_resolution + ".")
        else:
            raise ValueError("Unknown resize resolution!")
        if run_mode == 'one_by_one':
            print(Fore.YELLOW + "Please do not modify the output folder including its contents: " + out_path)
            print(Style.RESET_ALL)
            quality = 100 if steps_finished != steps_to_do else args.output_quality
            for img in tqdm(os.listdir(img_path)):
                im = cv.imread(os.path.join(img_path, img))
                im = resize.resize(im, resol, args.change_aspect_ratio, args.center_crop)
                cv.imwrite(os.path.join(out_path, img), im, [int(cv.IMWRITE_JPEG_QUALITY), quality])
            img_path = out_path
        else:
            images_ = []
            for img in tqdm(images):
                images_.append(resize.resize(img, resol, args.change_aspect_ratio, args.center_crop))
            images = images_

    if args.padding:
        # todo add padding
        print("\n--------------------------------")
        print("            Padding            ")
        print("--------------------------------\n")
        steps_finished += 1
        asp_ratio = args.target_aspect_ratio.split(':')
        if len(asp_ratio) != 2:
            raise ValueError("Wrong aspect ratio, form x:y is expected, got " + args.target_aspect_ratio + ' instead.')
        assert args.border >= 0, "Border must be positive!"
        asp_ratio = [float(asp_ratio[0]), float(asp_ratio[1])]
        if run_mode == "one_by_one":
            print(Fore.YELLOW + "Please do not modify the output folder including its contents: " + out_path)
            print(Style.RESET_ALL)
            quality = 100 if steps_finished != steps_to_do else args.output_quality
            for img in tqdm(os.listdir(img_path)):
                im = cv.imread(os.path.join(img_path, img))
                im = padding.padding(im, asp_ratio, args.border, True if args.padding_color == "white" else False)
                cv.imwrite(os.path.join(out_path, img), im, [int(cv.IMWRITE_JPEG_QUALITY), quality])
        else:
            images_ = []
            for img in tqdm(images):
                images_.append(
                    padding.padding(img, asp_ratio, args.border, True if args.padding_color == "white" else False))
            images = images_

    if run_mode == "together":
        for (img, img_name) in zip(images, image_files):
            cv.imwrite(os.path.join(out_path, img_name), img, [int(cv.IMWRITE_JPEG_QUALITY), args.output_quality])

    print(Fore.GREEN + f"Output finished at " + out_path + f' with {len(image_files)} images.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Argument for image-tools")
    
    # functions
    parser.add_argument("--add_watermark", action="store_true", default=False, help="Add watermarks for images")
    parser.add_argument("--resize", action="store_true", default=False, help="Resize for the images")
    parser.add_argument("--padding", action="store_true", default=False, help="Padding the images (for, e.g., instagram)")
    
    # general options
    parser.add_argument('--image_path', type=str, default='img/', help="Image folder (relative path)", required=False)
    parser.add_argument('--output_type', type=str, default='jpg', required=False)
    parser.add_argument('--output_path', type=str, default='output', required=False)
    parser.add_argument('--output_quality', type=int, default=100, help='quality of output images', required=False)
    
    # details for watermark
    parser.add_argument('--watermark_mode', type=str, default='preset_watermarks', help="watermarking modes", choices=("preset_watermarks", "text"))
    parser.add_argument('--no_apparent', action="store_true", help="No apparent watermark")
    parser.add_argument('--no_dark', action="store_true", help="No dark (hard-to-find) watermark")
    parser.add_argument('--wm_folder', type=str, default='wm/', required=False, help="Watermark folder, if you want to use preset images.")
    parser.add_argument('--apparent_wm_filename', type=str, default='apparent_wm.png', required=False)
    parser.add_argument('--apparent_wm_position', type=str, default='lower left', help="apparent watermark location")
    parser.add_argument('--apparent_wm_orientation', type=str, default='0', help="apparent watermark orientation, default: 0")
    parser.add_argument('--apparent_wm_scale', type=float, default=0.0625, help="apparent watermark relative scale, default: 0.0125")
    
    parser.add_argument('--dark_wm_filename', type=str, default='dark_wm.png', required=False)
    parser.add_argument('--dark_wm_scale', type=float, default=0.0625, required=False)
    parser.add_argument('--dark_wm_text', type=str, default='', required=False)
    parser.add_argument('--adaptative_dark', action="store_true", help="Adaptatively add dark watermarks to fit the background color")
    
    # details for resize/crop
    parser.add_argument('--resize_resolution', type=str, default='4k', choices=['4k', '2k', '2.5k', 'c4k', '8k'],
                        help='Resize resolution ratio')
    parser.add_argument('--change_aspect_ratio', action='store_true', help="Change aspect ratio to adapt to the resolution when input images do not fit that")
    parser.add_argument('--center_crop', action="store_true", help="Center crop to get the target resolution directly")
    
    # details for padding
    parser.add_argument('--border', type=float, default=0., help='padding on edges that has no expansion, in percentage')
    parser.add_argument('--target_aspect_ratio', type=str, default='1:1', help='Target aspect ratio')
    parser.add_argument('--padding_color', type=str, default='white', help='padding color', 
                        choices=("white", "black"))
    
    # running mode
    parser.add_argument('--run_mode', type=str, default='check_first', choices=("together", "one_by_one", "check_first"),
                        help="Deal with the images together or one-read-one-write or let the script determine itself.")
    
    args_ = parser.parse_args()
    main(args_)
