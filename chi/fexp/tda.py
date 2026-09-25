# Note: Going forward this should be implemented using Dashti's medtda package. But in the mean time, we can use this code, to implement the extraction, for now.
from gudhi import CubicalComplex
import numpy as np
import SimpleITK as sitk

from . import GetNewMethods

def bar_cleaner(barcode):
    if (np.size(barcode) > 0):
        return barcode[barcode[:,0]!=barcode[:,1]]
    else:
        return []

def GetPersStats(barcode,app=False,*p):
    barcode = bar_cleaner(barcode)
    if (np.size(barcode) > 0):
        # Average of Birth and Death of the barcode
        bc_av0, bc_av1 = np.mean(barcode, axis=0)
        # STDev of Birth and Death of the barcode
        bc_std0, bc_std1 = np.std(barcode, axis=0)
        # Median of Birth and Death of the barcode
        bc_med0, bc_med1 = np.median(barcode, axis=0)
        # Intercuartil range of births and death
        bc_iqr0, bc_iqr1 = np.subtract(*np.percentile(barcode, [75, 25], axis=0)) 
        # Range of births and deaths
        bc_r0, bc_r1=np.max(barcode, axis=0) - np.min(barcode, axis=0)
        # Percentiles of births and deaths
        bc_p10_0,bc_p10_1=np.percentile(barcode, 10, axis=0)
        bc_p25_0,bc_p25_1=np.percentile(barcode,25, axis=0)
        bc_p75_0,bc_p75_1=np.percentile(barcode, 75, axis=0)
        bc_p90_0,bc_p90_1=np.percentile(barcode, 90, axis=0)
        
        
        avg_barcodes = (barcode[:,1] + barcode[:,0])/2
        # Average of midpoints of the barcode
        bc_av_av = np.mean(avg_barcodes)
        # STDev of midpoints of the barcode
        bc_std_av = np.std(avg_barcodes)
        # Median of midpoints of the barcode
        bc_med_av = np.median(avg_barcodes)
        # Intercuartil range of midpoints
        bc_iqr_av = np.subtract(*np.percentile(avg_barcodes, [75, 25])) 
        # Range of midpoints
        bc_r_av = np.max(avg_barcodes) - np.min(avg_barcodes)
        # Percentiles of midpoints
        bc_p10_av = np.percentile(barcode, 10)
        bc_p25_av=np.percentile(barcode,25)
        bc_p75_av=np.percentile(barcode, 75)
        bc_p90_av=np.percentile(barcode, 90)
        
        diff_barcode = np.subtract([i[1] for i in barcode], [
                                   i[0] for i in barcode])
        diff_barcode = np.absolute(diff_barcode)
        # Average of the length of Bars
        bc_lengthAverage = np.mean(diff_barcode)
        # STD of length of Bars
        bc_lengthSTD = np.std(diff_barcode)
        # Median of length of Bars
        bc_lengthMedian = np.median(diff_barcode)
        # Intercuartil range of length of the bars
        bc_lengthIQR= np.subtract(*np.percentile(diff_barcode, [75, 25]))
        # Range of length of the bars
        bc_lengthR=np.max(diff_barcode) - np.min(diff_barcode)
        # Percentiles of lengths of the bars
        bc_lengthp10=np.percentile(diff_barcode, 10)
        bc_lengthp25=np.percentile(diff_barcode, 25)
        bc_lengthp75=np.percentile(diff_barcode, 75)
        bc_lengthp90=np.percentile(diff_barcode, 90)
        
        # Number of Bars
        bc_count = len(diff_barcode)
        # Persitent Entropy
        ent = GetNewMethods.Entropy()
        bc_ent = ent.fit_transform([barcode])
        
        bar_stats = np.array([bc_av0, bc_av1, bc_std0, bc_std1, bc_med0, bc_med1,
                              bc_iqr0, bc_iqr1, bc_r0, bc_r1, bc_p10_0, bc_p10_1, 
                              bc_p25_0, bc_p25_1, bc_p75_0, bc_p75_1, bc_p90_0, 
                              bc_p90_1, 
                              bc_av_av, bc_std_av, bc_med_av, bc_iqr_av, bc_r_av, bc_p10_av, 
                              bc_p25_av, bc_p75_av, bc_p90_av, bc_lengthAverage, bc_lengthSTD, 
                              bc_lengthMedian, bc_lengthIQR, bc_lengthR, bc_lengthp10,  
                              bc_lengthp25,  bc_lengthp75,  bc_lengthp90, bc_count, 
                              bc_ent[0][0]])
        if app == True:
            bar_stats = np.array([bc_av0, bc_std0, bc_med0, bc_iqr0, bc_r0, bc_p10_0, bc_p25_0, bc_p75_0, bc_p90_0,
                              bc_av1, bc_std1, bc_med1, bc_iqr1, bc_r1, bc_p10_1, bc_p25_1, bc_p75_1, bc_p90_1,
                              bc_av_av, bc_std_av, bc_med_av, bc_iqr_av, bc_r_av, bc_p10_av, bc_p25_av, bc_p75_av, bc_p90_av,
                              bc_lengthAverage, bc_lengthSTD, bc_lengthMedian, bc_lengthIQR, bc_lengthR, bc_lengthp10, bc_lengthp25, bc_lengthp75, bc_lengthp90,
                              bc_count, bc_ent[0][0]])
    else:
        bar_stats = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0,0, 0])

    bar_stats[~np.isfinite(bar_stats)] = 0

    return bar_stats

def compute_ph(img, max_dim=1):
    pbs = []

    # cub_filtration = CubicalComplex(dimensions=data.shape, top_dimensional_cells=data.flatten('F'))
    cub_filtration = CubicalComplex(top_dimensional_cells=img)
    cub_filtration.persistence()

    for i in range(max_dim + 1):  
        ph_temp = cub_filtration.persistence_intervals_in_dimension(i)
        ph_temp = ph_temp[~np.isinf(ph_temp).any(axis=1),:]
        pbs.append(ph_temp)

    return pbs

def make_barcods(img, msk, bg_value=-1000):
    ct_array = sitk.GetArrayFromImage(img)
    mask_array = sitk.GetArrayFromImage(msk)
    ct_array[mask_array == 0] = bg_value
    
    if ct_array.any():
        pbs = compute_ph(ct_array, max_dim=2)
    else:
        pbs = []
        print(f'Image ROI is empty!\n')

    return pbs
        

FEATURE_NAMES=['Births_Mean', 'Births_STD', 'Births_Median', 'Births_IQR', 'Births_Range', 'Births_P10', 
                      'Births_P25', 'Births_P75', 'Births_P90', 'Deaths_Mean', 'Deaths_STD', 'Deaths_Median', 
                      'Deaths_IQR', 'Deaths_Range', 'Deaths_P10', 'Deaths_P25', 'Deaths_P75', 'Deaths_P90', 
                      'Midpoints_Mean', 'Midpoints_STD', 'Midpoints_Median', 'Midpoints_IQR', 'Midpoints_Range', 
                      'Midpoints_P10', 'Midpoints_P25', 'Midpoints_P75', 'Midpoints_P90', 'Lifespans_Mean', 
                      'Lifespans_STD', 'Lifespans_Median', 'Lifespans_IQR', 'Lifespans_Range', 'Lifespans_P10', 
                      'Lifespans_P25', 'Lifespans_P75', 'Lifespans_P90', 'Count', 'Entropy']

def vectorize_barcode(pbs):
    all_results = {}
    for i, pd in enumerate(pbs):
        features = GetPersStats(pd)
        results = dict(zip([f"dim({i})_{f}" for f in FEATURE_NAMES], features))
        all_results.update(results)

    return all_results


class TDAExtractor:
    def __init__(self, conf):
        self.conf = conf
    
    def execute(self, im, msk):
        pds = make_barcods(im, msk, bg_value=self.conf['TDA']['bg_value'])
        vec = vectorize_barcode(pds)

        return vec