import radiomics, radiomics.featureextractor
import SimpleITK as sitk
from tqdm import tqdm
import pandas
import os, os.path

import argparse

def make_2d_extractor(fname):
    extractor = radiomics.featureextractor.RadiomicsFeatureExtractor()
    extractor.loadParams(fname)

    extractor.enableFeatureClassByName('shape2D', enabled=False)
    extractor.enableFeatureClassByName('shape', enabled=False)
    return extractor

def make_3d_extractor(fname):
    extractor = radiomics.featureextractor.RadiomicsFeatureExtractor()
    extractor.enableFeatureClassByName('shape2D', enabled=False)
    extractor.loadParams(fname)
    return extractor

class Processor:
    def __init__(self, extractors):
        self.extractors = extractors

    def process_row(self, row):
        index, row = row
        im = row['Image']
        msk = row['Mask']

        im = sitk.ReadImage(im)
        msk = sitk.ReadImage(msk)

        ress = {k: extractor.execute(im, msk) for k, extractor in self.extractors.items()}

        def add_to_row(x):
            dct = row.to_dict()
            dct.update(x)
            return dct

        ress = {k: add_to_row(r) for k, r in ress.items()}

        return index, ress

    def tabulate_results(self, results):
        output = {}
        for index, ress in results:
            for extractor, features in ress.items():
                output.setdefault(extractor, {})[index] = features

        output = {k: pandas.DataFrame.from_dict(v, orient='index') for k, v in output.items()}

        return output

def parse_confs(conf):
    cases = {}
    for c in conf:
        name, fil = c.split(':')
        cases[name] = make_3d_extractor(fil)
    return cases

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', required=True, nargs='+')
    #parser.add_argument("--conf3d", required=True)
    #parser.add_argument("--conf2d", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--jobs", default=1, type=int)

    args = parser.parse_args()

    table = pandas.read_csv(args.dataset, index_col=0)

    extractors = parse_confs(args.conf)
    #extractors = {"2d": make_2d_extractor(args.conf2d), "3d": make_3d_extractor(args.conf3d)}
    processor = Processor(extractors)

    if args.jobs == 1:
        results = processor.tabulate_results(map(processor.process_row, tqdm(table.iterrows(), total=table.shape[0])))
    else:
        import multiprocessing as mp
        with mp.Pool(processes=args.jobs) as pool:
            results = processor.tabulate_results(pool.map(processor.process_row, tqdm(table.iterrows(), total=table.shape[0])))

    dsname, _ = os.path.splitext(os.path.basename(args.dataset))
    for name, result in results.items():
        output_name = f"{dsname}_{name}.csv"
        result.to_csv(output_name)

if __name__=="__main__": main()
    






