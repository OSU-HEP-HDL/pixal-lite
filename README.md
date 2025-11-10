 ```
  ██████╗ ██╗██╗  ██╗ █████╗ ██╗         ██╗     ██╗████████╗███████╗
  ██╔══██╗██║╚██╗██╔╝██╔══██╗██║         ██║     ██║╚══██╔══╝██╔════╝
  ██████╔╝██║ ╚███╔╝ ███████║██║         ██║     ██║   ██║   █████╗
  ██╔═══╝ ██║ ██╔██╗ ██╔══██║██║         ██║     ██║   ██║   ██╔══╝  
  ██║     ██║██╔╝ ██╗██║  ██║███████╗    ███████╗██║   ██║   ███████╗
  ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝    ╚══════╝╚═╝   ╚═╝   ╚══════╝
                    PIXAL-LITE – Validation Module
```

This is an automated PIXAL validation framework that is to be used via a Docker image. 

## How It Works
PIXAL-Lite is hosted on swarm-worker-2 in the cluster. This validation framework uses a two packages, inotify and polling, to monitor a directory called `/validation`. When the Visual Inspection 
test starts, it creates sub-directories in `/validation` that correspond to the type of component that is being imaged. When the image taking process is complete, the photos are automatically 
transferred to the cluster to two locations. 

* The test results folder found in: `/mnt/images/itk_testing/<component>/<serial_number>/VISUAL_INSPECTION/`
* The validation directory: `/mnt/images/machine_learning/validation/<component>/<type_image1>/`

Once all sub-directories in `/validation` contain an image, PIXAL-Lite validation is triggered on the directory. Once validation is complete, the results are stored in `/results/<component>/<serial_number>/<type_image1>/<results>` 
and the `/validation` directory is cleared for the next test. 

## How To Place Nominal Models

The three directories that PIXAL-Lite uses are:

* `/mnt/images/machine_learning/validation/`
* `/mnt/images/machine_learning/results/`
* `/mnt/images/machine_learning/models/`

A component can have anywhere between 2 to 12 models associated with it since one model is trained on each type of image. Each model directory should follow a specific naming convention. For example, the `R0_DATA_FLEX_F1` only has its front and back photographed. So there are two models associated with it and have a specific and obvious name. In each model directory, it needs the metadata for the model and the model itself. 
```
/models/
  └── R0_DATA_FLEX_F1
       ├── R0_DATA_FLEX_F1_F
       |     ├── /metadata
       |     |     ├── model_training.yaml
       |     |     ├── paths.yaml
       |     |     ├── plotting.yaml
       |     |     ├── preprocessing.yaml
       |     |     └── testModel.yaml
       |     └──  /model
       |            └── testModel.h5
       └── R0_DATA_FLEX_F1_B
            ├── ...
```

The sub-directory names should be obvious, however, the only directory names that are required to be specifically named for the PIXAL-Lite workflow to output correctly are the component names. The names that should not change are:

* COUPLED_RING_R01
* INTERMEDIATE_RING
* L0_BARREL_DATA_FLEX_F1
* L0_BARREL_DATA_FLEX_F2
* L0_BARREL_DATA_FLEX_F3
* L0_BARREL_DATA_FLEX_F4
* L0_BARREL_POWER_FLEX_F1
* L0_BARREL_POWER_FLEX_F2
* L1_BARREL_DATA_FLEX
* L1_BARREL_POWER_FLEX
* QUAD_MODULE_Z_RAY_FLEX
* QUAD_RING_R1
* R0_DATA_FLEX_F1
* R0_DATA_FLEX_F2
* R0_DATA_FLEX_F3
* R0_POWER_T_F1
* R0_POWER_JUMPER_F2
* R05_DATA_FLEX_F1
* R05_DATA_FLEX_F2
* R0_Triplet_Data_F1
* TYPE0_TO_PP0_F1
* TYPE0_TO_PP0_F2

If you need to add or change anything on this list, edit the function `extract_component_name()` in `/pixal/modules/config_loader.py`

The directories are created within the hdl_webpage repository. You will need to add to this list as you add component models to pixa-lite. The function can be found in `/hdl_webpage/utils/proc.py`.
```
    # Default mapping (fill in more image types as needed)
    if component_validation_dirs is None:
        component_validation_dirs = {
            "R0_DATA_FLEX_F1": ("R0_DATA_FLEX_F1_F", "R0_DATA_FLEX_F1_B"),
            "R0_DATA_FLEX_F2": ("R0_DATA_FLEX_F2_F", "R0_DATA_FLEX_F2_B"),
            "R0_DATA_FLEX_F3": ("R0_DATA_FLEX_F3_F", "R0_DATA_FLEX_F3_B"),
            "R05_DATA_FLEX_F1": (),
            "R05_DATA_FLEX_F2": (),
            "R0_POWER_T_F1": (),
            "R0_POWER_JUMPER_F2": (),
            "TYPE0_TO_PP0_F1": (),
            "TYPE0_TO_PP0_F2": (),
            "L0_BARREL_POWER_FLEX": (),
            "L0_BARREL_DATA_FLEX": (),
            "L1_BARREL_DATA_FLEX_F1": (),
            "L1_BARREL_DATA_FLEX_F2": (),
            "L1_BARREL_DATA_FLEX_F3": (),
            "L1_BARREL_DATA_FLEX_F4": (),
            "L1_BARREL_POWER_FLEX_F1": (),
            "L1_BARREL_POWER_FLEX_F2": (),
            "QUAD_MODULE_Z_RAY_FLEX": (),
            "QUAD_RING_R1": (),
            "COUPLED_RING_R01": (),
            "INTERMEDIATE_RING": (),
        }

```

## Results

The reulst that are output from validation consists of mutlple metric plots, however, only one plot will be shown on the UI. The results for our `R0_DATA_FLEX_F1` example are output as such:
`/results/R0_DATA_FLEX_F1/<serial_number>/R0_DATA_FLEX_F1/R0_DATA_FLEX_F1_F/`

This folder is structured as such:
```
/results/.../R0_DATA_FLEX_F1_F/
  ├── out.npz
  ├── mse_overlay_0.png
  ├── /preprocessing
  |     └── /preprocessed_images
  |           └── ...
  └── /validation
        ├── anomaly_detection_curves_image_0.png
        ├── anomaly_overlay_0.png
        ├── combined channel_loss_histogram.png
        ├── combined_distribution_log.png
        ├── pixel_loss_histogram.png
        ├── pixel_loss_log_histogram.png
        ├── pixel_prediction.png
        ├── prediction_distribution_log.png
        └── truth_distribution_log.png
```

Where only the `mse_overlay_0.png` is shown on the UI.

