import argparse
import os, os.path

from joblib import Parallel, delayed
import pandas
import radiomics, radiomics.featureextractor
import SimpleITK as sitk

import numpy as np

#def make_2d_extractor(fname):
#    extractor = radiomics.featureextractor.RadiomicsFeatureExtractor()
#    extractor.loadParams(fname)
#
#    extractor.enableFeatureClassByName('shape2D', enabled=False)
#    extractor.enableFeatureClassByName('shape', enabled=False)
#    return extractor

def make_3d_extractor(fname):
    extractor = radiomics.featureextractor.RadiomicsFeatureExtractor()
    extractor.enableFeatureClassByName('shape2D', enabled=False)
    extractor.loadParams(fname)
    return extractor

class Label:
    @staticmethod
    def mask_equals(msk, label):
        return msk == label

    @staticmethod
    def mask_greaterthan(msk, label):
        return msk > label

    @staticmethod
    def mask_lessthan(msk, label):
        return msk < label

    def __init__(self, label, rule, prefix):
        self.label = label
        self.rule = rule
        self.prefix=prefix

    def apply(self, msk):
        return self.rule(msk, self.label)

    @classmethod
    def integer(cls, label):
        return cls(label, cls.mask_equals, "")

    @classmethod
    def lessthan(cls, label):
        return cls(label, cls.mask_lessthan, "lt")

    @classmethod
    def greaterthan(cls, label):
        return cls(label, cls.mask_greaterthan, "gt")

    def str(self, formatstr):
        intpart = formatstr.format(self.label)
        return self.prefix+intpart

def read_single_label(label):
    if label.startswith(">"):
        return Label.greaterthan(int(label[1:]))
    elif label.startswith("<"):
        return Label.lessthan(int(label[1:]))
    else:
        return Label.integer(int(label))

def read_label(label):
    if isinstance(label, str):
        if ',' in label:
            labels = [read_single_label(x) for x in label.split(',')]
        else:
            labels = [read_single_label(label)]
    else:
        labels = [Label.integer(int(label))]


    return labels

def get_labels_as_mask(seg, labels):
    assert len(labels) > 0
    msk = labels[0].apply(seg) # Initialize mask with first label
    for lab in labels[1:]:
        msk += lab.apply(seg)

    return msk

def format_labels(labels, intformat=":02d", joiner='-'):
    formatstr = f"{{{intformat}}}"

    strs = [l.str(formatstr) for l in labels]
    return joiner.join(strs)
    

def resample_mask_before_extraction(img, msk):
    rif = sitk.ResampleImageFilter()
    rif.SetReferenceImage(img)
    rif.SetInterpolator(sitk.sitkNearestNeighbor)

    return rif.Execute(msk)
# Note: This antialiasing code is borrowed from med-imagetools resampling code!
# https://github.com/bhklab/med-imagetools/blob/04bb636829a27489fa7a53e0dfb87c2e205b6619/src/imgtools/transforms/functional.py#L4

def prep_image_for_resample(
    image: sitk.Image,
    spacing: float | list[float] | np.ndarray,
    anti_alias_sigma: float | list[float] | None = None
) -> sitk.Image:
    """Resample an image to a new spacing with optional transform.

    Resamples the input image using the specified spacing, computing a new
    image size to maintain the original spatial extent unless explicitly set
    via output_size. A transformation can be applied during resampling, and
    Gaussian smoothing is used for anti-aliasing when downsampling.

    Parameters
    ----------
    image : sitk.Image
        The SimpleITK image to be resampled.
    spacing : float | list[float] | np.ndarray
        The desired spacing for each axis. A single float applies to all
        dimensions, while a sequence specifies spacing per axis. Use 0 for any
        axis to retain its original spacing.
    anti_alias_sigma : float | list[float] | None, optional
        The standard deviation for the Gaussian smoothing kernel. If not
        provided, it is automatically computed.

    Returns
    -------
    sitk.Image
        The filtered image, ready for resample.
    """

    original_spacing = np.array(image.GetSpacing())

    if isinstance(spacing, (float, int)):
        new_spacing = np.repeat(spacing, len(original_spacing)).astype(
            np.float64
        )
    else:
        spacing = np.asarray(spacing)
        new_spacing = np.where(spacing == 0, original_spacing, spacing)

    downsample = new_spacing > original_spacing
    if downsample.any():
        if not anti_alias_sigma:
            # sigma computation adapted from scikit-image
            # https://github.com/scikit-image/scikit-image/blob/master/skimage/transform/_warps.py
            anti_alias_sigma = list(
                np.maximum(1e-11, (original_spacing / new_spacing - 1) / 2)
            )
        sigma = np.where(downsample, anti_alias_sigma, 1e-11)
        image = sitk.SmoothingRecursiveGaussian(image, sigma)


    return image


class Processor:
    def __init__(self, extractors, dataset_root=None, image_column="Image", mask_column="Mask", label_column="MaskLabel", default_label=1, dump_preprocessed=False, dump_dir=None, resample_mask_before_extraction=False, skip_errors=False, use_antialiasing=False, extract_2d=False, slice_column='slice'):
        self.extractors = extractors
        self.dataset_root = dataset_root

        self.image_column=image_column
        self.mask_column=mask_column
        self.label_column=label_column
        self.default_label=default_label
        self.dump_preprocessed = dump_preprocessed
        self.dump_dir = dump_dir
        self.resample_mask_before_extraction=resample_mask_before_extraction
        self.skip_errors = skip_errors
        self.use_antialiasing = use_antialiasing
        self.extract_2d = extract_2d
        self.slice_column = slice_column
        

    def get_path(self, p):
        if self.dataset_root is None:
            return p
        else:
            return os.path.join(self.dataset_root, p)

    def read_image_and_mask(self, row):
        im = row[self.image_column]
        msk = row[self.mask_column]
        impath = im
        mskpath = msk

        label = row.get(self.label_column, self.default_label)
        labels = read_label(label)


        im = sitk.ReadImage(self.get_path(im))
        seg = sitk.ReadImage(self.get_path(msk))
        msk = get_labels_as_mask(seg, labels)

        return impath, mskpath, im, msk, labels

    def dump_process(self, row):
        index, row = row
        print(index, row[self.mask_column])

        impath, mskpath, im, msk, labels = self.read_image_and_mask(row)

        try:
            ress = {k: self.execute_extraction(im, msk, extractor, lambda i, m: extractor.loadImage(i, m, generalInfo=None, **extractor.settings.copy())) for k, extractor in self.extractors.items()}
        except:
            print("Error was in row", index, row)
            print("Image:", row[self.image_column])
            print("Mask:", row[self.mask_column])
            if not self.skip_errors:
                raise
            else:
                return

        def make_unique(patha, pathb):
            name = os.path.basename(patha).replace(".nii.gz", "")
            dir = os.path.dirname(patha)
            nameb = os.path.basename(pathb).replace(".nii.gz", "")
            nameo = "_".join(name, nameb)
            proposed = os.path.join(dir, nameo)
            assert not os.path.exists(proposed)
            return proposed



        result_rows = {}
        label = row.get(self.label_column, self.default_label)
        for conf, (lim, lmsk) in ress.items():
            conf_name = os.path.basename(conf).replace(".yaml", "")
            label_num = format_labels(labels)
            root = os.path.join(self.dump_dir, conf_name, label_num)
            newimpath = f"img_{index:05d}.nii.gz"
            newmskpath = f"msk_{index:05d}.nii.gz"
            imoutpath = os.path.join(root, newimpath)
            mskoutpath = os.path.join(root, newmskpath)
            os.makedirs(os.path.dirname(imoutpath), exist_ok=True)
            os.makedirs(os.path.dirname(mskoutpath), exist_ok=True)
            print("saving", imoutpath, mskoutpath)
            sitk.WriteImage(lim, imoutpath)
            sitk.WriteImage(lmsk, mskoutpath)

            out = row.to_dict()
            out[self.image_column] = os.path.relpath(imoutpath, self.dump_dir)
            out[self.mask_column] = os.path.relpath(mskoutpath, self.dump_dir)
            out[self.label_column] = 1
            out["OriginalMaskLabel"] = label

            result_rows[conf] = out

        return index, result_rows

    def execute_extraction(self, im, msk, extractor, extractor_method):
        if self.use_antialiasing:
            target_spacing = extractor.settings['resampledPixelSpacing']
            current_spacing = im.GetSpacing()
            actual_target = np.where(target_spacing, target_spacing, current_spacing)
            im = prep_image_for_resample(im, actual_target)

        if self.resample_mask_before_extraction:
            msk = resample_mask_before_extraction(im, msk)

        return extractor_method(im, msk)

    def process_chunk_2d(self, imp, mskp, rows):
        # Get image and mask
        tmp = {self.image_column: imp, self.mask_column: mskp}
        impath, mskpath, im, msk, labels = self.read_image_and_mask(tmp)

        all_results = {}
        for index, row in rows.iterrows():
            sl = row[self.slice_column]
            slim = im[:,:,sl]
            slmsk = msk[:,:,sl]
            try:
                ress = {k: self.execute_extraction(slim, slmsk, extractor, extractor.execute) for k, extractor in self.extractors.items()}
            except Exception as e:
                print("Error was in row", index, row)
                print("Image:", row[self.image_column])
                print("Mask:", row[self.mask_column])
                if not self.skip_errors:
                    raise RuntimeError(f"Error in row {index}, {row}. Image: {row[self.image_column]}, Mask: {row[self.mask_column]}") from e
                else:
                    import traceback
                    traceback.print_exc()
                    all_results[index] = None

            def add_to_row(x):
                dct = row.to_dict()
                dct.update(x)
                return dct

            ress = {k: add_to_row(r) for k, r in ress.items()}

            all_results[index] = ress
        return all_results




    def process_row(self, row):
        if self.dump_preprocessed:
            return self.dump_process(row)
        index, row = row
        print(index, row[self.mask_column])

        impath, mskpath, im, msk, labels = self.read_image_and_mask(row)

        try:
            ress = {k: self.execute_extraction(im, msk, extractor, extractor.execute) for k, extractor in self.extractors.items()}
        except Exception as e:
            print("Error was in row", index, row)
            print("Image:", row[self.image_column])
            print("Mask:", row[self.mask_column])
            if not self.skip_errors:
                raise RuntimeError(f"Error in row {index}, {row}. Image: {row[self.image_column]}, Mask: {row[self.mask_column]}") from e
            else:
                import traceback
                traceback.print_exc()
                return index, None

        def add_to_row(x):
            dct = row.to_dict()
            dct.update(x)
            return dct

        ress = {k: add_to_row(r) for k, r in ress.items()}

        return index, ress

    @staticmethod
    def tabulate_results(results):
        output = {}
        for index, ress in results:
            if ress is None:
                continue
            for extractor, features in ress.items():
                output.setdefault(extractor, {})[index] = features

        output = {k: pandas.DataFrame.from_dict(v, orient='index') for k, v in output.items()}

        return output

    @staticmethod
    def tabulate_results_2d(results):
        all_results = {k:v for r in results for k,v in r.items()}
        output = {}
        for index, ress in all_results.items():
            if ress is None:
                continue
            for extractor, features in ress.items():
                output.setdefault(extractor, {})[index] = features

        output = {k: pandas.DataFrame.from_dict(v, orient='index') for k, v in output.items()}

        return output

def parse_confs(conf):
    cases = {}
    for c in conf:
        cases[c] = make_3d_extractor(c)
    return cases

# TODO load conf files from resource files!
import time
class TicToc:
    def __init__(self, scaler=1e9):
        self.last = None
        self.scaler = scaler
        self.tic()
    
    def tic(self):
        self.last = time.perf_counter_ns()
    
    def toc(self):
        diff = time.perf_counter_ns() - self.last
        return diff / self.scaler

def do_execute(args, ix, row):
    extractors = parse_confs(args.conf)
    processor = Processor(extractors, dataset_root=args.dataset_root, image_column=args.image_column, mask_column=args.mask_column, label_column=args.label_column, default_label=args.use_label, dump_preprocessed=args.dump_preprocessed, dump_dir=args.dump_dir, resample_mask_before_extraction=args.resample_mask_before_extraction, skip_errors=args.skip_errors, use_antialiasing=args.use_antialiasing,
                          extract_2d=args.extract_2d, slice_column=args.slice_column)
    return processor.process_row((ix, row))

def do_execute_2d(args, imp, mskp, rows):
    extractors = parse_confs(args.conf)
    processor = Processor(extractors, dataset_root=args.dataset_root, image_column=args.image_column, mask_column=args.mask_column, label_column=args.label_column, default_label=args.use_label, dump_preprocessed=args.dump_preprocessed, dump_dir=args.dump_dir, resample_mask_before_extraction=args.resample_mask_before_extraction, skip_errors=args.skip_errors, use_antialiasing=args.use_antialiasing,
                          extract_2d=args.extract_2d, slice_column=args.slice_column)
    return processor.process_chunk_2d(imp, mskp, rows)

def main():
    tt = TicToc()
    parser = argparse.ArgumentParser(prog="python -m chi.fexp", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--conf', required=True, nargs='+',
                        help="A list of pyradiomics yaml configuration files (can be 1)")
    parser.add_argument('--output', required=True, nargs='+',
                        help="A list of output feature csv files for each specified configuration")
    parser.add_argument("--dataset", required=True,
                        help="A csv file specifying the images, masks, and labels for feature extraction")
    parser.add_argument("--dataset_root", required=False, default=None,
                        help="The root directory of the dataset, to which all paths in the data set file are relative")
    parser.add_argument("--jobs", default=1, type=int,
                        help="The number of parallel jobs to use. Default is 1 (serial)")
    parser.add_argument("--image_column", default="Image",
                        help="The name of the column specifying the image files.")
    parser.add_argument("--mask_column", default="Mask",
                        help="The name of the column specifying the mask file paths.")
    parser.add_argument("--label_column", default="MaskLabel",
                        help="The name of the column specifying the label(s) to use.")
    parser.add_argument("--dump_preprocessed", action='store_true',
                        help="Trigger the program to output preprocessed images, rather than imaging features")
    parser.add_argument("--dump_dir",
                        help="When outputting preprocessed images, this specifies the root path for output.")
    parser.add_argument("--use_label", default=1, type=int,
                        help="If no label_column is present in the data set, this argument specifies the default label to use for all images")
    parser.add_argument("--start", default=-1, type=int,
                        help="Offset execution, processing the rows starting at the given index")
    parser.add_argument("--count", default=-1, type=int,
                        help="Limit execution to this number of rows, starting from the start index.")
    parser.add_argument("--resample_mask_before_extraction", action="store_true",
                        help="This prevents certain rare errors..")
    parser.add_argument("--skip_errors", action="store_true",
                        help="Skip (and log) errors.")
    parser.add_argument("--use_antialiasing", action="store_true",
                        help="Use antialiasing when downsampling")
    parser.add_argument("--extract_2d", action="store_true",
                        help="2D extraction mode")
    parser.add_argument("--slice_column", default='slice',
                        help="slice to consider for slice index in 2D mode.")



    args = parser.parse_args()
    print(args.conf)
    print(args.output)

    table = pandas.read_csv(args.dataset)

    if args.start > -1:
        assert args.count > -1
        istart = args.start
        iend = args.start + args.count
        iend = min(iend, table.shape[0])

        print("Batch mode, computing rows", istart, "to", iend)

        table = table.iloc[istart:iend]


    extractors = parse_confs(args.conf)
    outputs = dict(zip(args.conf, args.output))
    #processor = Processor(extractors, dataset_root=args.dataset_root, image_column=args.image_column, mask_column=args.mask_column, label_column=args.label_column, default_label=args.use_label)


    print("Time to load up:", tt.toc())

    

    if not args.extract_2d:
        results = Parallel(n_jobs=args.jobs, verbose=10)(delayed(do_execute)(args, ix, row) for ix, row in table.iterrows())
        print("Merging results")
        results = Processor.tabulate_results(results)
    else:
        results = Parallel(n_jobs=args.jobs, verbose=10)(delayed(do_execute_2d)(args, imp, mskp, rows) for (imp, mskp), rows in table.groupby([args.image_column, args.mask_column]))
        print("Merging results")
        results = Processor.tabulate_results_2d(results)
    #if args.jobs == 1:
    #    results = processor.tabulate_results(map(processor.process_row, tqdm(table.iterrows(), total=table.shape[0])))
    #else:
    #    import multiprocessing as mp
    #    with mp.Pool(processes=args.jobs) as pool:
    #        results = processor.tabulate_results(pool.map(processor.process_row, table.iterrows()))

    print("Saving results")
    for name, result in results.items():
        output = outputs[name]
        result.to_csv(output, index=False)

    print("Total run time", tt.toc())

if __name__=="__main__": main()
    






