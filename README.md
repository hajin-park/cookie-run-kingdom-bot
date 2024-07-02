# Cookie Run Kingdom Resource Production Farming Bot

YOLOv10 finetuned model on 63 unique Cookie Run Kingdom production objects.
Custom dataset labeled and prepared using [Roboflow](https://roboflow.com/).

## Requirements

-   BlueStacks emulator
-   Virtualization enabled

## Installation

-   `conda env create -f environment.yml -n cookie_run_kingdom_bot`

## Setup

-   Create a `.env` in the repository's root directory.
-   Fill in the following environment variables with your Roboflow credentials
    `ROBOFLOW_API_KEY=""`\
    `ROBOFLOW_WORKSPACE_ID=""`\
    `ROBOFLOW_MODEL_ID=""`
