Example config file: 

```
gaussian {
    name = "shuttle";                                         (name of dataset)
    fpath = "../resources/data/shuttle.csv";                  (location of source dataset file)
    qpath = "../resources/data/shuttle.csv";                  (location of query dataset file)
    exact_path = "../resources/exact/shuttle_gaussian.txt";   (location of query dataset file)
    kernel = "gaussian";           (kernel type, either "gaussian" or "exp")
    d = "9";                       (dimension of dataset)
    n = "43500";                   (# data points in source dataset)
    m = "43500";                   (# queries; 
                                    if m == n, compute density for all points in dataset,
                                    if m < n, take m random points from source as queries.)
    h = "1";                       (bandwidth parameter)
    bw_const = "false";            (if true, set bandwidth = h. 
                                    if false, for Exponential kernels, set bw=h * Scott's factor,
                                    for Gaussian kernels, bw=sqrt(2) * h * Scott's factor)
    ignore_header = "false";       (whether or not to ignore the first line of the dataset)
    start_col = "0";               (column index to start reading data from)
    end_col = "8";                 (column index to stop reading data from)
    eps = "0.5";                   (1+/-epsilon accuracy)                      
    tau = "0.0001";                (Minimum density)
    beta = "0.5";                  (Parameter for HBE's hashing scheme; use 0.5 as default.)
    sample_ratio = "4.5";          (Controls the wall-clock runtime of HBE and RS. See main/BatchBenchmark.cpp. )
}
```

By default, the data ingestion code assumes that the dataset has been preprocessed so that the **standard deviation for each column is 1**. This means that the bandwidth parameter is a constant for each column. If this is not the case, the bandwidth should be scaled with the standard deviation for each column (set bandwidth using Bandwidth::getBandwidth() instead of  Bandwidth::useConstant()[https://github.com/kexinrong/rehashing/blob/master/hbe/utils/DataIngest.h#L80]).
