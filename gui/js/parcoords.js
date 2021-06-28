/* global IApi, CosmoScout */

(() => {
  class Parcoords {
    constructor(rootId, name, csv, callback) {
      const root     = document.getElementById(rootId);
      const template = CosmoScout.gui.loadTemplateContent("volumeRendering-parcoords");
      root.appendChild(template);
      root.parcoords = this;

      this.id                    = rootId;
      this.parcoords             = root.querySelector(".parcoords");
      this.parcoordsControls     = root.querySelector(".parcoordsControls");
      this.parcoordsParent       = this.parcoordsControls.parentNode;
      this.parcoordsUndockButton = root.querySelector(".parcoordsUndockButton");
      this.popout           = CosmoScout.gui.loadTemplateContent("volumeRendering-parcoordsPopout");
      this.popout.innerHTML = this.popout.innerHTML.replace("%NAME%", name);
      document.getElementById("cosmoscout").appendChild(this.popout);
      this.popoutContent = this.popout.querySelector(".window-content");
      this.popoutWrapper = this.popout.querySelector(".window-wrapper");

      this.parcoordsUndockButton.addEventListener("click", () => { this.undock(); });
      this.popout.querySelector(`[data-action="close"]`)
          .addEventListener("click", () => { this.dock(); });

      this.exportLocation = this.parcoordsControls.querySelector(".parcoordsExportLocation");
      this.parcoordsControls.querySelector(".parcoordsExport").addEventListener("click", () => {
        CosmoScout.callbacks.parcoords.exportBrushState(
            this.exportLocation.value, JSON.stringify(this.exportBrushState()));
      });
      this.importSelect = this.parcoordsControls.querySelector(".parcoordsImportSelect");
      this.parcoordsControls.querySelector(".parcoordsImport").addEventListener("click", () => {
        CosmoScout.callbacks.parcoords.importBrushState(this.importSelect.value, this.id);
      });

      this.brushMin         = this.parcoordsControls.querySelector(".parcoordsBrushMin");
      this.brushMax         = this.parcoordsControls.querySelector(".parcoordsBrushMax");
      this.activeBrushLabel = this.parcoordsControls.querySelector(".parcoordsActiveBrushLabel");
      this.activeBrush      = "";

      this.brushMin.addEventListener("change", () => {
        let max = this.brushMax.value;
        if (max == "") {
          max = this.pc.dimensions()[this.activeBrush].yscale.domain()[1];
        }
        this.pc.brushExtents({[this.activeBrush]: [this.brushMin.value, max]});
      });
      this.brushMax.addEventListener("change", () => {
        let min = this.brushMin.value;
        if (min == "") {
          min = this.pc.dimensions()[this.activeBrush].yscale.domain()[0];
        }
        this.pc.brushExtents({[this.activeBrush]: [min, this.brushMax.value]});
      });

      this.heightOffset = 120;

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
      const boundCallback = callback.bind(this);
      this.pc.on("brush", (brushed, args) => {
        boundCallback(brushed, args);
        if (args) {
          this._updateMinMax(args.axis);
        }
      });
      this.pc.on("brushend", (brushed, args) => {
        if (args) {
          this._updateMinMax(args.axis);
        }
      });
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

      this.setHeight(200);
    }

    exportBrushState() {
      const brushExtents = this.pc.brushExtents();
      Object.keys(brushExtents)
          .forEach(e => { brushExtents[e] = brushExtents[e].selection.scaled.reverse(); });
      return brushExtents;
    }

    importBrushState(json) {
      const brushExtents = JSON.parse(json);
      this.pc.brushExtents(brushExtents);
    }

    setAvailableBrushStates(availableFiles) {
      let options = "";
      availableFiles.forEach((file) => { options += `<option>${file}</option>`; });
      this.importSelect.innerHTML = options;
      $(this.importSelect).selectpicker();
      $(this.importSelect).selectpicker("refresh");
    }

    undock() {
      this.parcoordsUndockButton.hidden = true;
      this.popout.classList.add("visible");
      this.parcoordsControls = this.parcoordsParent.removeChild(this.parcoordsControls);
      this.popoutContent.appendChild(this.parcoordsControls);
      this.setHeight(this.popoutWrapper.clientHeight - this.heightOffset);
    }

    dock() {
      this.parcoordsUndockButton.hidden = false;
      this.parcoordsControls            = this.popoutContent.removeChild(this.parcoordsControls);
      this.parcoordsParent.appendChild(this.parcoordsControls);
      this.setHeight(200);
    }

    setHeight(height) {
      // BrushExtents are deleted when resizing, so they have to be carried over manually
      const brushExtents = this.exportBrushState();
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

    _updateMinMax(dimension) {
      CosmoScout.callbacks.volumeRendering.setPathlineActiveScalar(
          "point_" + dimension.replace("_start", "").replace("_end", ""));
      this.activeBrushLabel.innerText = dimension;
      this.activeBrush                = dimension;
      this.brushMin.disabled          = false;
      this.brushMax.disabled          = false;
      if (this.pc.brushExtents().hasOwnProperty(dimension)) {
        this.brushMin.value = this.pc.brushExtents()[dimension].selection.scaled[1];
        this.brushMax.value = this.pc.brushExtents()[dimension].selection.scaled[0];
      } else {
        this.brushMin.value = "";
        this.brushMax.value = "";
      }
    }
  }

  /**
   * Parallel coordinates editor API
   */
  class ParcoordsApi extends IApi {
    /**
     * @inheritDoc
     */
    name = "parcoords";

    /**
     * List of created parallel coordinates editors
     * @type {Parcoords[]}
     */
    _editors = [];

    /**
     * List of available parcoord brushes files
     * @type {string[]}
     */
    _availableFiles = [];

    /**
     * @inheritDoc
     */
    init() {
    }

    /**
     * TODO
     */
    create(rootId, name, csv, callback) {
      const parcoords = new Parcoords(rootId, name, csv, callback);
      parcoords.setAvailableBrushStates(this._availableFiles);
      this._editors.push(parcoords);

      if (CosmoScout.callbacks.parcoords !== undefined) {
        CosmoScout.callbacks.parcoords.getAvailableBrushStates();
      }
      return parcoords;
    }

    /**
     * Sets the list of available brush states for all created editors.
     *
     * @param availableFilesJson {string[]} List of filenames
     */
    setAvailableBrushStates(availableFilesJson) {
      this._availableFiles = JSON.parse(availableFilesJson);
      this._editors.forEach((editor) => { editor.setAvailableBrushStates(this._availableFiles); });
    }

    /**
     * Adds one filename to the list of available files.
     * Afterwards updates the available files for all editors.
     *
     * @param filename {string} Name of the file to be added
     */
    addAvailableBrushState(filename) {
      if (!this._availableFiles.includes(filename)) {
        let files = this._availableFiles;
        files.push(filename);
        files.sort();
        this.setAvailableBrushStates(JSON.stringify(files));
      }
    }

    /**
     * Loads a brush state to one editor.
     *
     * @param json {string} json string describing the brush state
     * @param editorId {number} ID of the editor for which the state should be loaded
     */
    loadBrushState(json, editorId) {
      this._editors.find(e => e.id == editorId).importBrushState(json);
    }
  }

  CosmoScout.init(ParcoordsApi);
})();
