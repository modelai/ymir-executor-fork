import os
import imagesize



def _annotation_path_for_image(image_path: str, annotations_dir: str) -> str:
    # replace dir
    annotation_path = image_path.replace('/in/assets', annotations_dir, 1)
    # replace ext
    annotation_path = os.path.splitext(annotation_path)[0] + '.txt'
    return annotation_path


def _convert_annotations(index_file_path: str, dst_annotations_dir: str) -> None:
    """
    read all images and annotations in `index_file_path`, change annotations format, write to `annotations_dir`

    `annotations_dir` should exists
    """
    with open(index_file_path, 'r') as f:
        files = f.readlines()
        files = [each.strip() for each in files]

    N = len(files)
    for i, each_img_anno_path in enumerate(files):
        if i % 1000 == 0:
            print(f"converted {i}/{N} image annotations")

        # each_imgpath: asset path
        # each_txtfile: annotation path
        each_imgpath, each_txtfile = each_img_anno_path.split()

        img_w, img_h = imagesize.get(each_imgpath)

        with open(each_txtfile, 'r') as f:
            txt_content = f.readlines()
            txt_content = [each.strip() for each in txt_content]
        output_list = []

        for each_line in txt_content:
            each_line = [int(each) for each in each_line.split(",")[0:5]]

            cls, xmin, ymin, xmax, ymax, *_ = each_line
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(img_w, xmax)
            ymax = min(img_h, ymax)
            xcenter = (xmin + xmax) / 2
            ycenter = (ymin + ymax) / 2
            bbox_w = xmax - xmin
            bbox_h = ymax - ymin

            if bbox_w < 10 or bbox_h < 10:
                # too small, ignored
                continue

            xcenter /= img_w
            ycenter /= img_h
            bbox_w /= img_w
            bbox_h /= img_h
            output_str = f"{cls} {xcenter} {ycenter} {bbox_w} {bbox_h}"
            output_list.append(output_str)

        # write output_list
        dst_txtfile = _annotation_path_for_image(image_path=each_imgpath, annotations_dir=dst_annotations_dir)
        os.makedirs(os.path.dirname(dst_txtfile), exist_ok=True)
        with open(dst_txtfile, 'w') as f:
            for each_str in output_list:
                f.write(f"{each_str}\n")


def _create_image_index_file(src_index_path: str, dst_index_path: str) -> None:
    with open(src_index_path, 'r') as f:
        lines = f.read().splitlines()

    with open(dst_index_path, 'w') as f:
        for line in lines:
            f.write(f"{line.split()[0]}\n")


if __name__ == "__main__":
    _create_image_index_file(src_index_path='/in/train-index.tsv', dst_index_path='/out/train-index-assets.tsv')
    _create_image_index_file(src_index_path='/in/val-index.tsv', dst_index_path='/out/val-index-assets.tsv')
    _convert_annotations(index_file_path='/in/train-index.tsv', dst_annotations_dir='/out/tmp_labels')
    _convert_annotations(index_file_path='/in/val-index.tsv', dst_annotations_dir='/out/tmp_labels')
