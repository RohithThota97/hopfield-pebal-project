# Ablation Study Report
Generated: 2025-09-23 20:58:58

## Executive Summary

- Optimal memory size: 40000
- Optimal beta value: 4
- Best diversity method: random

## Performance Metrics


## Detailed Results

```json
{
  "feature_extractor": {
    "(1, 3, 256, 512)": {
      "input_shape": [
        1,
        3,
        256,
        512
      ],
      "output_shape": [
        1,
        512
      ],
      "mean": 0.00028009869856759906,
      "std": 0.048290882259607315,
      "has_nan": false,
      "has_inf": false
    },
    "(2, 3, 512, 1024)": {
      "input_shape": [
        2,
        3,
        512,
        1024
      ],
      "output_shape": [
        2,
        512
      ],
      "mean": 0.0003107353113591671,
      "std": 0.0482807457447052,
      "has_nan": false,
      "has_inf": false
    },
    "(1, 3, 768, 1536)": {
      "input_shape": [
        1,
        3,
        768,
        1536
      ],
      "output_shape": [
        1,
        512
      ],
      "mean": 0.0002917706733569503,
      "std": 0.0482761450111866,
      "has_nan": false,
      "has_inf": false
    }
  },
  "memory_builder": {
    "{'id_memory_size': 10000, 'aux_memory_size': 5000, 'diversity_method': 'cluster'}": {
      "id_shape": [
        10000,
        128
      ],
      "aux_shape": [
        5000,
        128
      ],
      "warnings": [
        "ID memory padded from 5 to 10000",
        "AUX memory padded from 5 to 5000"
      ],
      "id_norm": 11.29633903503418,
      "aux_norm": 11.28610610961914
    },
    "{'id_memory_size': 20000, 'aux_memory_size': 10000, 'diversity_method': 'farthest_point'}": {
      "id_shape": [
        20000,
        128
      ],
      "aux_shape": [
        10000,
        128
      ],
      "warnings": [
        "ID memory padded from 5 to 20000",
        "AUX memory padded from 5 to 10000"
      ],
      "id_norm": 11.296966552734375,
      "aux_norm": 11.279170989990234
    },
    "{'id_memory_size': 30000, 'aux_memory_size': 15000, 'diversity_method': 'hybrid'}": {
      "id_shape": [
        30000,
        128
      ],
      "aux_shape": [
        15000,
        128
      ],
      "warnings": [
        "ID memory padded from 5 to 30000",
        "AUX memory padded from 5 to 15000"
      ],
      "id_norm": 11.290729522705078,
      "aux_norm": 11.289118766784668
    }
  },
  "energy_functions": {
    "border_beta_8": {
      "mean": -1.504815697669983,
      "std": 0.0055886320769786835,
      "min": -1.5183753967285156,
      "max": -1.4941577911376953
    },
    "border_beta_16": {
      "mean": -1.5063146352767944,
      "std": 0.015622604638338089,
      "min": -1.5430164337158203,
      "max": -1.4710750579833984
    },
    "border_beta_32": {
      "mean": -1.54664146900177,
      "std": 0.11879715323448181,
      "min": -2.116300582885742,
      "max": -1.3926010131835938
    },
    "border_beta_64": {
      "mean": -2.180643081665039,
      "std": 0.9800274968147278,
      "min": -7.13885498046875,
      "max": -1.3866806030273438
    },
    "inference_scores": {
      "mean": 0.7517987489700317,
      "std": 0.32292309403419495,
      "positive_ratio": 0.9799999594688416
    }
  },
  "hopfield_boosting": {
    "{'beta_sampling': 32, 'num_boosting_iters': 1}": {
      "weight_entropy": 0.6219922498917254,
      "batch_diversity": 0.6138945711728843,
      "ood_loss": 0.7765971817842738
    },
    "{'beta_sampling': 64, 'num_boosting_iters': 3}": {
      "weight_entropy": 0.9469792002905488,
      "batch_diversity": 0.6642645113085083,
      "ood_loss": 0.3356620055071048
    },
    "{'beta_sampling': 128, 'num_boosting_iters': 5}": {
      "weight_entropy": 0.3496079369812556,
      "batch_diversity": 0.8669697300055675,
      "ood_loss": 0.0416029380990629
    }
  },
  "memory_sizes": {
    "10000": {
      "id_shape": [
        10000,
        128
      ],
      "aux_shape": [
        5000,
        128
      ],
      "build_time_s": 13.618427276611328,
      "warnings": [
        "ID memory padded from 50 to 10000",
        "AUX memory padded from 50 to 5000"
      ],
      "diversity_score": 0.9597110375761986,
      "coverage_score": -43.511199951171875
    },
    "20000": {
      "id_shape": [
        20000,
        128
      ],
      "aux_shape": [
        10000,
        128
      ],
      "build_time_s": 13.350397825241089,
      "warnings": [
        "ID memory padded from 50 to 20000",
        "AUX memory padded from 50 to 10000"
      ],
      "diversity_score": 0.996771291596815,
      "coverage_score": -42.94317626953125
    },
    "30000": {
      "id_shape": [
        30000,
        128
      ],
      "aux_shape": [
        15000,
        128
      ],
      "build_time_s": 13.081845045089722,
      "warnings": [
        "ID memory padded from 50 to 30000",
        "AUX memory padded from 50 to 15000"
      ],
      "diversity_score": 0.9949426464736462,
      "coverage_score": -44.47926330566406
    },
    "40000": {
      "id_shape": [
        40000,
        128
      ],
      "aux_shape": [
        20000,
        128
      ],
      "build_time_s": 13.391835689544678,
      "warnings": [
        "ID memory padded from 50 to 40000",
        "AUX memory padded from 50 to 20000"
      ],
      "diversity_score": 1.010239865630865,
      "coverage_score": -43.10523223876953
    },
    "50000": {
      "id_shape": [
        50000,
        128
      ],
      "aux_shape": [
        25000,
        128
      ],
      "build_time_s": 13.37472128868103,
      "warnings": [
        "ID memory padded from 50 to 50000",
        "AUX memory padded from 50 to 25000"
      ],
      "diversity_score": 1.0084910159930587,
      "coverage_score": -43.982582092285156
    }
  },
  "diversity_methods": {
    "random": {
      "memory_used_mb": 262.009765625,
      "diversity_score": 1.0265649035573006,
      "separation": 0.0060781959211331045
    },
    "farthest_point": {
      "memory_used_mb": 262.009765625,
      "diversity_score": 1.0039834696799517,
      "separation": 0.005984299449944275
    }
  },
  "beta_values": {
    "4": {
      "border_energy_mean": -19.52719497680664,
      "border_energy_std": 16.1900634765625,
      "score_mean": 0.32697340846061707,
      "score_std": 25.37586212158203,
      "score_range": [
        -77.99427032470703,
        96.85111236572266
      ]
    },
    "8": {
      "border_energy_mean": -39.0477294921875,
      "border_energy_std": 32.51795196533203,
      "score_mean": 0.6554690599441528,
      "score_std": 50.839752197265625,
      "score_range": [
        -155.99609375,
        194.24050903320312
      ]
    },
    "16": {
      "border_energy_mean": -78.08477020263672,
      "border_energy_std": 65.12196350097656,
      "score_mean": 1.3139314651489258,
      "score_std": 101.72775268554688,
      "score_range": [
        -311.9922180175781,
        388.7734375
      ]
    },
    "32": {
      "border_energy_mean": -156.15576171875,
      "border_energy_std": 130.2985382080078,
      "score_mean": 2.6315836906433105,
      "score_std": 203.4802703857422,
      "score_range": [
        -623.9844360351562,
        777.6063232421875
      ]
    },
    "64": {
      "border_energy_mean": -312.301025390625,
      "border_energy_std": 260.6272277832031,
      "score_mean": 5.266294479370117,
      "score_std": 406.97186279296875,
      "score_range": [
        -1247.9688720703125,
        1555.2144775390625
      ]
    }
  },
  "memory_subset_sizes": {
    "10000": {
      "mean_time_ms": 2.8374294750392437,
      "std_time_ms": 0.002649163763074744,
      "energy_mean": -128.26368713378906,
      "energy_std": 105.17621612548828
    }
  },
  "performance_profile": {
    "feature_extraction": {
      "mean_ms": 4.0956927463412285,
      "std_ms": 0.033733566538442505,
      "min_ms": 4.064932931214571,
      "max_ms": 4.146642051637173
    },
    "projection": {
      "mean_ms": 4.108691215515137,
      "std_ms": 0.002904175850566446,
      "min_ms": 4.104082006961107,
      "max_ms": 4.112821072340012
    },
    "border_energy": {
      "mean_ms": 4.284380283206701,
      "std_ms": 0.021210744020959967,
      "min_ms": 4.259683191776276,
      "max_ms": 4.323496948927641
    },
    "inference_score": {
      "mean_ms": 4.234747588634491,
      "std_ms": 0.02426311962943237,
      "min_ms": 4.217042122036219,
      "max_ms": 4.282786976546049
    }
  },
  "memory_usage": {
    "peak_memory_mb": 293.93603515625,
    "used_memory_mb": 262.0,
    "id_memory_mb": 9.765625,
    "aux_memory_mb": 4.8828125
  },
  "scalability": {
    "1": {
      "time_ms": 1.0076680531104405,
      "time_per_sample_ms": 1.0076680531104405,
      "memory_mb": 64.0
    },
    "2": {
      "time_ms": 1.8164409945408504,
      "time_per_sample_ms": 0.9082204972704252,
      "memory_mb": 128.0
    },
    "4": {
      "time_ms": 3.359827989091476,
      "time_per_sample_ms": 0.839956997272869,
      "memory_mb": 256.0
    },
    "8": {
      "time_ms": 6.367667267719905,
      "time_per_sample_ms": 0.7959584084649881,
      "memory_mb": 512.0
    }
  },
  "quality_metrics": {
    "10000": {
      "separation_ratio": 0.007780686593429976,
      "center_distance": 0.1749429553747177,
      "id_spread": 11.239789962768555,
      "aux_spread": 11.244465827941895
    }
  },
  "efficiency": {
    "batch_scaling": {
      "times": [
        0.004294633865356445,
        0.006850719451904297,
        0.013996124267578125,
        0.03306937217712402,
        0.0739445686340332
      ],
      "memory": [
        718.07763671875,
        988.080078125,
        1524.08251953125,
        2596.08740234375,
        4740.09716796875
      ],
      "throughput": [
        232.84872036862265,
        291.9401405999861,
        285.793404197329,
        241.9156903599778,
        216.37829925261005
      ]
    },
    "memory_scaling": {
      "times": [
        3.8953542709350586,
        7.398891448974609,
        10.667181015014648,
        13.946008682250977,
        17.499375343322754
      ],
      "memory": [
        7.32421875,
        14.6484375,
        21.97265625,
        29.296875,
        36.62109375
      ]
    }
  }
}
```
