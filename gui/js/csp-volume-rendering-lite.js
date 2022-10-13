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
      CosmoScout.gui.initSlider("volumeRendering.setResolution", 32, 2048, 32, [256]);
      CosmoScout.gui.initSliderRange("volumeRendering.setSamplingRate",
        { "min": 0.001, "25%": 0.01, "50%": 0.1, "75%": 1, "max": 10 }, [0.005]);
      CosmoScout.gui.initSlider("volumeRendering.setAOSamples", 0, 10, 1, [4]);
      CosmoScout.gui.initSlider("volumeRendering.setMaxRenderPasses", 1, 100, 1, [10]);
      CosmoScout.gui.initSlider("volumeRendering.setDensityScale", 1, 200, 0.1, [10]);
      CosmoScout.gui.initSliderRange("volumeRendering.setPathlineSize", { "min": 0.001, "25%": 0.01, "50%": 0.1, "75%": 1, "max": 10 }, [0.001]);

      CosmoScout.gui.initSlider("volumeRendering.setSunStrength", 0, 10, 0.1, [1]);
      CosmoScout.gui.initSlider("volumeRendering.setAmbientStrength", 0, 1, 0.01, [0.5]);

      CosmoScout.gui.initSlider("volumeRendering.setRotationYaw", 0, 360, 1, [0]);
      CosmoScout.gui.initSlider("volumeRendering.setRotationPitch", 0, 360, 1, [0]);
      CosmoScout.gui.initSlider("volumeRendering.setRotationRoll", 0, 360, 1, [0]);

      this.progressBar = document.getElementById("volumeRendering.progressBar");

      this.transferFunctionEditor = CosmoScout.transferFunctionEditor.create(
        document.getElementById("volumeRendering.tfEditor"), this.setTransferFunction,
        { fitToData: true });

      this._enablePathlinesParcoordsCheckbox =
        document.querySelector(`[data-callback="volumeRendering.setEnablePathlinesParcoords"]`);

      this.playing = false;
    }

    enableSettingsSection(section, enable = true) {
      document.getElementById(`headingVolume-${section}`).parentElement.hidden = !enable;
    }

    initParcoords(volumeData, pathlinesData) {
      const me = this;
      this._parcoordsVolume = CosmoScout.parcoords.create(
        "volumeRendering-parcoordsVolume", "Volume", volumeData, function (brushed, args) {
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
          "Pathlines", pathlinesData, function (brushed, args) {
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
        const timestep = Number(this.timestepSlider.noUiSlider.get());
        brushState = Object.keys(brushState).reduce((accum, k) => {
          let state = accum;
          state[k + "_start"] = brushState[k];
          state[k + "_end"] = brushState[k];
          return state;
        }, {});
        brushState["InjectionStepId_start"] = {
          "selection": { "scaled": [timestep + 0.5, timestep - 9.5] }
        };
      }
      CosmoScout.callbacks.volumeRendering.setPathlinesScalarFilters(JSON.stringify(brushState));
    }

    copyParcoords(fromParcoords, toParcoords) {
      let brushState = fromParcoords.exportBrushState();
      brushState = Object.keys(brushState).reduce((accum, k) => {
        let state = accum;
        state[k + "_start"] = brushState[k];
        state[k + "_end"] = brushState[k];
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

    loadTransferFunction(path) {
      CosmoScout.callbacks.transferFunctionEditor.importTransferFunction(
        path, this.transferFunctionEditor.id);
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
  }

  CosmoScout.init(VolumeRenderingApi);
})();
