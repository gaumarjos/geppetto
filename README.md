# GePpetto

A homemade gpx viewer able to create climb gradient plots.

    python app.py

Then open [http://localhost:8050/](http://localhost:8050/) in a web browser.

## Requirements

Developed with Python 10.0.0. Install required modules with:

    pip install -r requirements.txt

## Project structure

* `app.py` is the web app
* `geppetto.py` is where all the math is
* `geppetto_obj.py` is an object implementation of `geppetto.py` that is not maintained because it doesn't work well
  with the Dash app
* `tracks/` is where all the `.gpx` and `.fit` files are

## Known bugs

1. The app interface looks very bad
2. For some unknown reason, the map plot doesn't update when a new file is loaded or a portion of it is selected from the
  elevation plot. It looks like the trace is cached somewhere. It's the only plot that doesn't react to updates. Posted 
  here https://github.com/plotly/dash/issues/1152. This
  makes no sense because
    * its center updates correctly
    * datapoints are updated correctly ![](docs/map_update_bug.png)
    * if replaced with a scatter, it updates correctly
    * the other minimap in the gradient plot updates as expected
   Posted in https://github.com/plotly/dash/issues/1152