/* global IApi, CosmoScout */

(() => {
  class Parcoords {
    constructor(rootId, name, csv, callback) {
      const root     = document.getElementById(rootId);
      const template = CosmoScout.gui.loadTemplateContent("volumeRendering-parcoords");
      root.appendChild(template);

      this.id                    = rootId;
      this.parcoords             = root.querySelector(".parcoords");
      this.parcoordsScroller     = root.querySelector(".parcoordsScroller");
      this.parcoordsParent       = this.parcoordsScroller.parentNode;
      this.parcoordsUndockButton = root.querySelector(".parcoordsUndockButton");
      this.popout           = CosmoScout.gui.loadTemplateContent("volumeRendering-parcoordsPopout");
      this.popout.innerHTML = this.popout.innerHTML.replace("%NAME%", name);
      document.getElementById("cosmoscout").appendChild(this.popout);
      this.popoutContent = this.popout.querySelector(".window-content");
      this.popoutWrapper = this.popout.querySelector(".window-wrapper");

      this.parcoordsUndockButton.addEventListener("click", () => { this.undock(); });
      this.popout.querySelector(`[data-action="close"]`)
          .addEventListener("click", () => { this.dock(); });

      this.heightOffset = 80;

      this.data                                    = d3.csvParse(csv);
      this.data                                    = d3.shuffle(this.data);
      const width                                  = 100 * this.data.columns.length;
      root.querySelector(".parcoords").style.width = `${width}px`;
      this.pc                                      = ParCoords()(`#${this.id} .parcoords`);
      this.pc.data(this.data)
          .width(width)
          .height(200)
          .color("#99f")
          .alpha(0.05)
          .mode("queue")
          .rate(50)
          .render()
          .brushMode("1D-axes")
          .interactive();
      this.pc.on("brush", callback.bind(this));
      this.parcoords.querySelectorAll("g.tick line, path.domain")
          .forEach(e => {e.style.stroke = "var(--cs-color-text)"});

      const target        = this.popoutWrapper;
      const config        = {attributes: true};
      this.resizeObserver = new MutationObserver((mutations, observer) => {
        for (const mut of mutations) {
          if (mut.type === "attributes") {
            if (mut.attributeName === "style") {
              this.setHeight(target.clientHeight - this.heightOffset);
            }
          }
        }
      });
      this.resizeObserver.observe(target, config);
    }

    exportBrushes() {
      const brushExtents = this.pc.brushExtents();
      Object.keys(brushExtents)
          .forEach(e => { brushExtents[e] = brushExtents[e].selection.scaled.reverse(); });
      return brushExtents;
    }

    undock() {
      this.parcoordsUndockButton.hidden = true;
      this.popout.classList.add("visible");
      this.parcoordsScroller = this.parcoordsParent.removeChild(this.parcoordsScroller);
      this.popoutContent.appendChild(this.parcoordsScroller);
      this.setHeight(this.popoutWrapper.clientHeight - this.heightOffset);
    }

    dock() {
      this.parcoordsUndockButton.hidden = false;
      this.parcoordsScroller            = this.popoutContent.removeChild(this.parcoordsScroller);
      this.parcoordsParent.appendChild(this.parcoordsScroller);
      this.setHeight(200);
    }

    setHeight(height) {
      // BrushExtents are deleted when resizing, so they have to be carried over manually
      const brushExtents = this.exportBrushes();
      // The range of the yscale is used to determine the height with which the axes will be
      // rendered. It seems, as if the range is not updated automatically when setting the
      // parcoords height, so instead it is updated manually here.
      Object.values(this.pc.dimensions()).forEach((d) => {
        d.yscale.range([height - this.pc.state.margin.top - this.pc.state.margin.bottom + 1, 1]);
      });
      // Apparently brushMode has to be switched back and forth once, or else the brushing will
      // not work after a resize
      this.pc.height(height).render().brushMode("angular").brushMode("1D-axes").brushExtents(
          brushExtents);
      this.parcoords.style.height = `${height}px`;
      this.parcoords.querySelectorAll("g.tick line, path.domain")
          .forEach(e => {e.style.stroke = "var(--cs-color-text)"});
    }
  }

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

      this._parcoordsVolume = new Parcoords(
          "volumeRendering-parcoordsVolume", "Volume", csvData, function(brushed, args) {
            CosmoScout.callbacks.volumeRendering.setVolumeScalarFilters(
                JSON.stringify(this.pc.brushExtents()));
          });
      this._parcoordsPathlines = new Parcoords(
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
