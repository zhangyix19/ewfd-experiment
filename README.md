# Data Organization

## File Tree

```plaintext
data
├── README.md
├── captured
│   └── undefend
│       └── crawl
│           ├── batch-x
│           │   └── url-y
│           │       ├── <label>.png
│           │       ├── label
│           │       ├── tcp.pcap
│           │       └── time
│           └── ip
├── extracted
│   └── undefend.npz
├── truncated
│   └── undefend.npz
├── cell_level
│   └── undefend.npz
├── defended
│   └── undefend
│       └── front.npz
└── wang
    └── undefend
        ├── x
        └── x-y

```

## Directory Contents

**captured** [file tree]: pcaps, logs, and pngs organized as file tree above. For data captured by **our** data collection framework.

**extracted** [npz file]: contains packet traces(NDArray[Shape["* traces, * packets, [tick, dir, size] dims"], Float]) and labels(list[str]), extracted from **captured** dir.

**truncated** [npz file]: contains packet traces(NDArray[Shape["* traces, * packets, [tick, dir, size] dims"], Float]) and labels(list[str]), truncated by a shrehold of a 10s max 2-pkt interval for trace start and 12s for trace end.

**cell_level** [npz file]: contains packet traces(NDArray[Shape["* traces, * cells, [tick, dir] dims"], Float]) and labels(NDArray[Shape["* labels"],Int]), split the tcp packet with different size to Tor cell by 536 Bytes.

**defended** [npz file]: contains packet traces(NDArray[Shape["* traces, * cells, [tick, dir] dims"], Float]) and labels(NDArray[Shape["* labels"],Int]), cell level defended trace named by the defense algorithm.

## Data Flow

captured/\<datasetname\> --extract.py--> extracted/\<datasetname\>.npz --dataset.py.Dataset.truncate()--> truncated/\<datasetname\>.nzp --dataset.py.Dataset.to_cell_level()--> cell_level/\<datasetname\>.npz --dataset.py.Dataset.defend(defense)--> defend/\<datasetname\>/\<defensename\>.npz

# BEFORE RUN

## download img validator model

```shell
wget -i imgvalid/model/model.link -O model.pth
```

## install python dependency packages

```shell

```

# RUN
