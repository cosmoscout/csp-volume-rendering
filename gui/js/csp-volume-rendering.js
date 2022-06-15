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
      CosmoScout.gui.initSlider("volumeRendering.setAnimationSpeed", 1, 100, 1, [20]);
      CosmoScout.gui.initSlider("volumeRendering.setResolution", 32, 2048, 32, [256]);
      CosmoScout.gui.initSliderRange("volumeRendering.setSamplingRate",
          {"min": 0.001, "25%": 0.01, "50%": 0.1, "75%": 1, "max": 10}, [0.005]);
      CosmoScout.gui.initSlider("volumeRendering.setAOSamples", 0, 10, 1, [4]);
      CosmoScout.gui.initSlider("volumeRendering.setMaxRenderPasses", 1, 100, 1, [10]);
      CosmoScout.gui.initSlider("volumeRendering.setDensityScale", 1, 20, 0.1, [10]);
      CosmoScout.gui.initSlider("volumeRendering.setPathlineSize", 1, 20, 1, [1]);

      CosmoScout.gui.initSlider("volumeRendering.setSunStrength", 0, 10, 0.1, [1]);
      CosmoScout.gui.initSlider("volumeRendering.setAmbientStrength", 0, 1, 0.01, [0.5]);

      CosmoScout.gui.initSlider("volumeRendering.setRotationYaw", 0, 360, 1, [0]);
      CosmoScout.gui.initSlider("volumeRendering.setRotationPitch", 0, 360, 1, [0]);
      CosmoScout.gui.initSlider("volumeRendering.setRotationRoll", 0, 360, 1, [0]);

      // Trigger "setTimestep" callback on "update" event
      this.timestepSlider = document.querySelector(`[data-callback="volumeRendering.setTimestep"]`);
      this.timestepSlider.dataset.event = "update";
      this.timestepContainer = document.getElementById("volumeRendering.timestepContainer");
      this.animationSpeedContainer =
          document.getElementById("volumeRendering.animationSpeedContainer");
      this.playButton = document.getElementById("volumeRendering.play");

      this.progressBar = document.getElementById("volumeRendering.progressBar");

      this.transferFunctionEditor = CosmoScout.transferFunctionEditor.create(
          document.getElementById("volumeRendering.tfEditor"), this.setTransferFunction,
          {fitToData: true});

      this._enablePathlinesParcoordsCheckbox =
          document.querySelector(`[data-callback="volumeRendering.setEnablePathlinesParcoords"]`);

      this.playing = false;
    }

    enableSettingsSection(section, enable = true) {
      document.getElementById(`headingVolume-${section}`).parentElement.hidden = !enable;
    }

    initParcoords(volumeData, pathlinesData) {
      const me              = this;
      this._parcoordsVolume = CosmoScout.parcoords.create(
          "volumeRendering-parcoordsVolume", "Volume", volumeData, function(brushed, args) {
            CosmoScout.callbacks.volumeRendering.setVolumeScalarFilters(
                JSON.stringify(this.pc.brushExtents()));
            if (pathlinesData !== "" && !me._enablePathlinesParcoordsCheckbox.checked) {
              me.setPathlinesScalarFilters(this.pc.brushExtents(), true);
            }
          });

      if (pathlinesData !== "") {
        this._parcoordsPathlinesContainer =
            document.getElementById("volumeRendering-parcoordsPathlines");
        this._parcoordsPathlines = CosmoScout.parcoords.create("volumeRendering-parcoordsPathlines",
            "Pathlines", pathlinesData, function(brushed, args) {
              if (me._enablePathlinesParcoordsCheckbox.checked) {
                me.setPathlinesScalarFilters(this.pc.brushExtents());
              }
            });

        this._enablePathlinesParcoordsCheckbox.addEventListener("change", (e) => {
          if (e.target.checked) {
            this.setPathlinesScalarFilters(this._parcoordsPathlines.pc.brushExtents());
          } else {
            this.setPathlinesScalarFilters(this._parcoordsVolume.pc.brushExtents(), true);
            if (!this._parcoordsPathlines.docked) {
              this._parcoordsPathlines.dock();
            }
          }
          this._parcoordsPathlinesContainer.hidden = !e.target.checked;
        });

        const copyToPathlinesWrapper = document.createElement("div");
        copyToPathlinesWrapper.classList.add("row");
        copyToPathlinesWrapper.innerHTML = `
        <div class="col-12">
          <button class="waves-effect waves-light block btn glass text">Copy to pathlines</button>
        </div>
      `;
        copyToPathlinesWrapper.querySelector(".btn").addEventListener("click", () => {
          this.copyParcoords(this._parcoordsVolume, this._parcoordsPathlines);
        });
        this._parcoordsVolume.parcoordsControls.appendChild(copyToPathlinesWrapper);
      }
    }

    setPathlinesScalarFilters(brushState, fromVolume = false) {
      if (fromVolume) {
        const timestep                      = Number(this.timestepSlider.noUiSlider.get());
        brushState                          = Object.keys(brushState).reduce((accum, k) => {
          let state           = accum;
          state[k + "_start"] = brushState[k];
          state[k + "_end"]   = brushState[k];
          return state;
        }, {});
        brushState["InjectionStepId_start"] = {
          "selection": {"scaled": [timestep + 0.5, timestep - 9.5]}
        };
      }
      CosmoScout.callbacks.volumeRendering.setPathlinesScalarFilters(JSON.stringify(brushState));
    }

    copyParcoords(fromParcoords, toParcoords) {
      let brushState = fromParcoords.exportBrushState();
      brushState     = Object.keys(brushState).reduce((accum, k) => {
        let state           = accum;
        state[k + "_start"] = brushState[k];
        state[k + "_end"]   = brushState[k];
        return state;
      }, {});
      toParcoords.pc.brushExtents(brushState);
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
      this.playing = !this.playing;
      CosmoScout.callbacks.volumeRendering.setTimestepAnimating(this.playing);

      const playButtonIcon = $("#volumeRendering\\.play i");
      playButtonIcon.html(this.playing ?  "pause" : "play_arrow");

      if (this.playing) {
        this.nextTimestep();
      }
    }

    nextTimestep() {
      const timeStep = parseInt(this.timestepSlider.noUiSlider.get());

      const current = this.timesteps.findIndex(t => t == timeStep);
      const next = (current+1) % this.timesteps.length;
      const preload = (current+2) % this.timesteps.length;

      CosmoScout.callbacks.volumeRendering.preloadTimestep(this.timesteps[preload]);
      CosmoScout.gui.setSliderValue("volumeRendering.setTimestep", true, this.timesteps[next]);

      const speedSlider =
              document.querySelector(`[data-callback="volumeRendering.setAnimationSpeed"]`)
                  .noUiSlider;
      const delay = 1000.0 / parseInt(speedSlider.get());

      setTimeout(() => {
        if (this.playing) {
          this.nextTimestep();
        }
      }, delay);
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
     * If there is only one timestep, the timestep slider and the animation feature
     * will be disabled.
     *
     * @param timestepsJson {string} json string of a list of all available timesteps
     */
    setTimesteps(timestepsJson) {
      this.timesteps = JSON.parse(timestepsJson);

      if (this.timesteps.length < 2) {
        this.timestepContainer.hidden       = true;
        this.animationSpeedContainer.hidden = true;
        this.playButton.hidden              = true;
        return;
      }

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
      this.timestepSlider.noUiSlider.on("update", (values, handle, unencoded) => {
        if (this._parcoordsPathlines !== undefined) {
          this._parcoordsPathlines.pc.brushExtents(
              {"InjectionStepId_start": [unencoded - 9.5, unencoded + 0.5]});
          if (!this._enablePathlinesParcoordsCheckbox.checked) {
            this.setPathlinesScalarFilters(this._parcoordsVolume.pc.brushExtents(), true);
          }
        }
      });
    }

    loadTransferFunction(path) {
      CosmoScout.callbacks.transferFunctionEditor.importTransferFunction(
          path, this.transferFunctionEditor.id);
    }
  }

  CosmoScout.init(VolumeRenderingApi);
})();
