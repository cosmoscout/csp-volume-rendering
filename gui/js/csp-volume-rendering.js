/* global IApi, CosmoScout */

(() => {
  /**
   * Volume Rendering Api
   */
  class VolumeRenderingApi extends IApi {
    /**
     * @inheritDoc
     */
    name = 'volumeRendering';

    /**
     * @inheritDoc
     */
    init() {
      CosmoScout.gui.initInputs();
      CosmoScout.gui.initSlider("volumeRendering.setAnimationSpeed", 10, 1000, 10, [100]);
      CosmoScout.gui.initSlider("volumeRendering.setResolution", 32, 2048, 32, [256]);
      CosmoScout.gui.initSliderRange("volumeRendering.setSamplingRate",
          {"min": 0.001, "25%": 0.01, "50%": 0.1, "75%": 1, "max": 10}, [0.005]);
      CosmoScout.gui.initSlider("volumeRendering.setSunStrength", 0, 10, 0.1, [1]);
      CosmoScout.gui.initSlider("volumeRendering.setDensityScale", 0, 10, 0.1, [1]);
      CosmoScout.gui.initSlider("volumeRendering.setPathlineOpacity", 0, 1, 0.05, [1]);
      CosmoScout.gui.initSlider("volumeRendering.setPathlineSize", 1, 10, 1, [1]);

      // Trigger "setTimestep" callback on "update" event
      var timestepSlider = document.querySelector(`[data-callback="volumeRendering.setTimestep"]`);
      timestepSlider.dataset.event = "update";

      this.progressBar = document.getElementById("volumeRendering.progressBar");

      this.transferFunctionEditor = CosmoScout.transferFunctionEditor.create(
          document.getElementById("volumeRendering.tfEditor"), this.setTransferFunction,
          {fitToData: true});

      this._parcoordsVolume = CosmoScout.parcoords.create(
          "volumeRendering-parcoordsVolume", "Volume", csvData, function(brushed, args) {
            CosmoScout.callbacks.volumeRendering.setVolumeScalarFilters(
                JSON.stringify(this.pc.brushExtents()));
          });
      this._parcoordsPathlines = CosmoScout.parcoords.create(
          "volumeRendering-parcoordsPathlines", "Pathlines", csvData2, function(brushed, args) {
            CosmoScout.callbacks.volumeRendering.setPathlinesScalarFilters(
                JSON.stringify(this.pc.brushExtents()));
          });
    }

    /**
     * Sets the transfer function for this plugin's transfer function editor.
     *
     * @param transferFunction {string} A json string describing the transfer function
     */
    setTransferFunction(transferFunction) {
      const callback = CosmoScout.callbacks.find("volumeRendering.setTransferFunction");
      if (callback !== undefined) {
        callback(transferFunction);
      }
    }

    /**
     * Toggles automatic progression of the timestep slider.
     * If the automatic progression is enabled, the slider will move a certain amount of units per
     * second, set using the animation speed slider.
     */
    play() {
      const playButtonIcon = $("#volumeRendering\\.play i");
      if (this.playing) {
        clearInterval(this.playHandle);
        playButtonIcon.html("play_arrow");
        this.playing = false;
      } else {
        const timeSlider =
            document.querySelector(`[data-callback="volumeRendering.setTimestep"]`).noUiSlider;
        this.time       = parseInt(timeSlider.get());
        let prevNext    = this.time;
        this.playHandle = setInterval(() => {
          // reverse() changes the original array, so it is called twice to return the array to the
          // original state
          const last = this.timesteps.reverse().find(t => t <= parseInt(this.time));
          const next = this.timesteps.reverse().find(t => t > parseInt(this.time));
          if (!next) {
            this.play();
          }
          let preload = CosmoScout.callbacks.find("volumeRendering.preloadTimestep");
          if (next && next != prevNext && preload !== undefined) {
            preload(next);
            prevNext = next;
          }
          CosmoScout.gui.setSliderValue("volumeRendering.setTimestep", true, last);
          const speedSlider =
              document.querySelector(`[data-callback="volumeRendering.setAnimationSpeed"]`)
                  .noUiSlider;
          this.time += parseInt(speedSlider.get()) / 10;
        }, 100);
        playButtonIcon.html("pause");
        this.playing = true;
      }
    }

    /**
     * Updates the progress bar in the rendering section of the settings menu.
     *
     * @param progress {number} Current progress of the rendering process.
     *     Should be value between 0 and 1.
     * @param animate {bool} Sets whether the progress bar should show the new value instantly or if
     *     there should be a smooth transition.
     */
    setRenderProgress(progress, animate) {
      if (animate) {
        this.progressBar.classList.add('animated');
      } else {
        this.progressBar.classList.remove('animated');
      }

      this.progressBar.style.width = Math.round(progress * 100) + "%";
    }

    /**
     * Sets the available timesteps. The timestep slider will snap to these values.
     *
     * @param timestepsJson {string} json string of a list of all available timesteps
     */
    setTimesteps(timestepsJson) {
      this.timesteps = JSON.parse(timestepsJson);
      this.timesteps.sort((a, b) => a - b);
      var min   = this.timesteps[0];
      var max   = this.timesteps[this.timesteps.length - 1];
      var range = {};
      this.timesteps.forEach((t, i) => {
        var percent = ((t - min) / (max - min) * 100) + "%";
        if (t == min) {
          percent = "min";
        } else if (t == max) {
          percent = "max";
        }
        range[percent]    = [];
        range[percent][0] = t;
        if (t != max) {
          range[percent][1] = this.timesteps[i + 1] - t;
        }
      });
      CosmoScout.gui.initSliderRange("volumeRendering.setTimestep", range, [this.timesteps[0]]);
    }

    /**
     * Sets the range of the currently active scalar.
     *
     * @param xMin {number} Minimum scalar value
     * @param xMax {number} Maximum scalar value
     * @param newScalar {bool} Should be true, if the active scalar changed
     */
    setXRange(xMin, xMax, newScalar) {
      this.transferFunctionEditor.setData([xMin, xMax], newScalar);
    }

    loadTransferFunction(path) {
      CosmoScout.callbacks.transferFunctionEditor.importTransferFunction(
          path, this.transferFunctionEditor.id);
    }
  }

  CosmoScout.init(VolumeRenderingApi);
})();
