 ```
  ██████╗ ██╗██╗  ██╗ █████╗ ██╗         ██╗     ██╗████████╗███████╗
  ██╔══██╗██║╚██╗██╔╝██╔══██╗██║         ██║     ██║╚══██╔══╝██╔════╝
  ██████╔╝██║ ╚███╔╝ ███████║██║         ██║     ██║   ██║   █████╗
  ██╔═══╝ ██║ ██╔██╗ ██╔══██║██║         ██║     ██║   ██║   ██╔══╝  
  ██║     ██║██╔╝ ██╗██║  ██║███████╗    ███████╗██║   ██║   ███████╗
  ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝    ╚══════╝╚═╝   ╚═╝   ╚══════╝
                    PIXAL-LITE – Validation Module
```

This is a PIXAL validation framework that is to be used via a Docker image. 

The output is saved in the input folder of the visual inspection result directory.

```
<Serial_number>/
  └── <VISUAL_INSPECTION>/
      ├── <image_type1>/
      |    ├── logs/
      |    ├── metadata/
      |    ├── figures/
      |    │   ├── anomaly_overlay_*.png
      |    │   ├── pixel_loss_histogram.png
      |    │   └── ...
      |    └── aligned_metrics/
      └── <image_type2>
```


The model that the docker container should be stored on the servers with these files. The model folder contains the ML model. The metadata contains the saved .yaml config files that are generated after training. The reference folder should contain a single image of the component, taken from the directory `aligned_images`. This image is used for preprocessing the validating image. 
```
models/
  └── <component_model>/
      ├── <image_type1>
      |    ├── model/
      |    ├── metadata/
      |    └── reference/
      └── <image_type2>
```   

Here is an example of how to run the docker image. The `-v` arguments are mounting a drive. The /< component >/< serial_number >/VISUAL_INSPECTION/ need to be passed as arguements as PIXAL-lite uses the information to store the output properly. 
```
docker run --rm \
  --gpus all \
  -v $(pwd)/models/R0_DATA_FLEX_F1:/model \
  -v $(pwd)/itk_testing/R0_DATA_FLEX_F1/20UPIPG9000023/VISUAL_INSPECTION/:/R0_DATA_FLEX_F1/20UPIPG9000023/VISUAL_INSPECTION \
  pixal-lite validate -m /model -i /R0_DATA_FLEX_F1/20UPIPG9000023/VISUAL_INSPECTION 
```

Everything subject to change