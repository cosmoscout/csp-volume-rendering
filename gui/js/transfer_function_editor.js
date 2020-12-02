/* global IApi, CosmoScout */

// This transfer function editor is heavily based on the tf-editor component
// (https://github.com/VolumeRC/tf-editor). The code was modified to be usable without using web
// components. Copyright (c) 2016 Vicomtech-IK4. All rights reserved. This code may only be used
// under the Apache License 2.0 found at http://volumerc.github.io/tf-editor/LICENSE.txt The
// complete set of attributions may be found at http://volumerc.github.io/tf-editor/NOTICE Copies of
// the LICENSE.txt and NOTICE can be found next to this file as tf-editor_LICENSE.txt and
// tf-editor_NOTICE.

(() => {
  class TransferFunctionEditor {
    constructor(
        id, callback, {width = 400, height = 150, fitToData = false, numberTicks = 5} = {}) {
      this.id       = id;
      this.callback = callback;
      this.options = {width: width, height: height, fitToData: fitToData, numberTicks: numberTicks};

      this._createElements();
      this._initializeElements();
      this._drawChart();
    }

    setData(data) {
      this._data = data;
      this._updateScales();
      this._updateAxis();
    }

    _createElements() {
      // Axis scales
      this._xScale = d3.scale.linear();
      this._yScale = d3.scale.linear();

      // Area for the opacity map representation
      this._area = d3.svg.area();

      // Keep track of control points interaction
      this._dragged    = null;
      this._selected   = null;
      this._last_color = 'green';

      this._controlPoints = [];

      // Custom margins
      this._margin = {top: 10, right: 20, bottom: 25, left: 40};

      // Access the svg dom element
      this._svg = d3.select("#transferFunctionEditor\\.graph-" + this.id)
                      .attr("width", this.options.width)
                      .attr("height", this.options.height);
      this._width  = +this._svg.attr("width") - this._margin.left - this._margin.right;
      this._height = +this._svg.attr("height") - this._margin.top - this._margin.bottom;
    }

    _initializeElements() {
      CosmoScout.gui.initInputs();

      let extent = [0, 255];
      if (this.options.fitToData && this._data && this._data.length > 0) {
        extent = d3.extent(this._data);
      }
      this._xScale.rangeRound([0, this._width]).domain(extent);
      this._yScale.domain([0, 1]).range([this._height, 0]);
      if (this._controlPoints.length == 0) {
        this._controlPoints.push(
            {'x': extent[0], 'opacity': 0, 'color': '#0000FF', 'locked': true});
        this._controlPoints.push(
            {'x': extent[1], 'opacity': 1, 'color': '#FF0000', 'locked': true});
      }
      this._selected = this._controlPoints[1];
      this._area.x(d => { return this._xScale(d.x); })
          .y0(d => { return this._yScale(d.opacity); })
          .y1(this._height);

      // Access the color selector
      this._colorPicker = $("#transferFunctionEditor\\.colorPicker-" + this.id).get(0);
      this._colorPicker.picker.on("change", () => {
        this._selected.color = this._colorPicker.value;
        this._redraw();
      });
      // Export button listener
      $("#transferFunctionEditor\\.export-" + this.id).on("click", () => {
        CosmoScout.callbacks.transferFunctionEditor.exportTransferFunction(
            $("#transferFunctionEditor\\.exportLocation-" + this.id).val(), this.getJsonString());
      });
      // Import button listener
      $("#transferFunctionEditor\\.import-" + this.id).on("click", () => {
        CosmoScout.callbacks.transferFunctionEditor.importTransferFunction(
            $("#transferFunctionEditor\\.importSelect-" + this.id).val(), this.id);
      });

      // Lock button listener
      $("#transferFunctionEditor\\.colorLock-" + this.id).on("click", () => {
        if (this._controlPoints.some(point => point.locked && point !== this._selected)) {
          this._selected.locked = !this._selected.locked;
          this._redraw();
          this._updateLockButtonState();
        }
      });
    }

    _updateLockButtonState() {
      const colorLockButton = $("#transferFunctionEditor\\.colorLock-" + this.id + " i");
      if (this._selected.locked) {
        $(colorLockButton).html("lock");
      } else {
        $(colorLockButton).html("lock_open");
      }
    }

    // Perform the drawing
    _drawChart() {
      const g =
          this._svg.append("g")
              .attr("transform", "translate(" + this._margin.left + "," + this._margin.top + ")")
              .on("mouseleave", () => { this._mouseup(); });

      // Gradient definitions
      g.append("defs")
          .append("linearGradient")
          .attr("id", "transferFunctionEditor.gradient-" + this.id)
          .attr("gradientUnits", "objectBoundingBox")
          .attr("spreadMethod", "pad")
          .attr("x1", "0%")
          .attr("y1", "0%")
          .attr("x2", "100%")
          .attr("y2", "0%");

      // Draw control points
      g.append("path")
          .datum(this._controlPoints)
          .attr("class", "line")
          .attr("fill", "url(#transferFunctionEditor.gradient-" + this.id + ")");

      g.append("path").datum(this._controlPoints).attr("class", "line").attr("fill", "none");

      // Mouse interaction handler
      g.append("rect")
          .attr("y", -10)
          .attr("x", -10)
          .attr("width", this._width + 20)
          .attr("height", this._height + 20)
          .style("opacity", 0)
          .on("mousedown", () => { this._mousedown(); })
          .on("mouseup", () => { this._mouseup(); })
          .on("mousemove", () => { this._mousemove(); });

      // Draw axis
      const xTicks              = this._xScale.ticks(this.options.numberTicks);
      xTicks[xTicks.length - 1] = this._xScale.domain()[1];
      g.append("g")
          .attr("class", "axis axis--x")
          .attr("transform", "translate(0," + this._height + ")")
          .call(d3.svg.axis().scale(this._xScale).orient("bottom").tickValues(xTicks));

      g.append("g")
          .attr("class", "axis axis--y")
          .attr("transform", "translate(0, 0)")
          .call(d3.svg.axis().scale(this._yScale).orient("left").ticks(this.options.numberTicks));
    }

    // update scales with new data input
    _updateScales() {
      if (this.options.fitToData) {
        let dataExtent = [];
        if (this._data && this._data.length > 0) {
          dataExtent = d3.extent(this._data);
        }
        if (dataExtent[0] == dataExtent[1]) {
          dataExtent[1] += 1;
        }
        this._xScale.domain(dataExtent);
      } else {
        this._xScale.domain([0, 255]);
      }
    }

    // update the axis with the new data input
    _updateAxis() {
      const svg                 = d3.select("svg").select("g");
      const xTicks              = this._xScale.ticks(this.options.numberTicks);
      xTicks[xTicks.length - 1] = this._xScale.domain()[1];
      svg.selectAll(".axis.axis--x").call(d3.axisBottom(this._xScale).tickValues(xTicks));
    }

    _updateControlPoints(controlPoints) {
      const updateScale = d3.scale.linear()
                              .domain(d3.extent(controlPoints, point => point.x))
                              .range(this._xScale.domain());
      controlPoints.forEach(point => point.x = updateScale(point.x));
      this._controlPoints = controlPoints;
    }

    // Update the chart content
    _redraw() {
      if (!this._controlPoints.some(point => point.locked)) {
        this._selected.locked = !this._selected.locked;
        this._redraw();
        this._updateLockButtonState();
      }
      this._controlPoints.forEach((point, index) => {
        if (index == 0) {
          point.x = this._xScale.invert(0);
        } else if (index == this._controlPoints.length - 1) {
          point.x = this._xScale.invert(this._width);
        }

        if (!point.locked) {
          const right = this._controlPoints.slice(index, this._controlPoints.length)
                            .find(point => point.locked);
          const left = this._controlPoints.slice(0, index).reverse().find(point => point.locked);
          if (left && right) {
            point.color =
                d3.interpolateRgb(left.color, right.color)((point.x - left.x) / (right.x - left.x));
          } else if (left) {
            point.color = left.color;
          } else if (right) {
            point.color = right.color;
          }
        }
      });

      const svg = d3.select("#transferFunctionEditor\\.graph-" + this.id).select("g");
      svg.selectAll("path.line").datum(this._controlPoints).attr("d", this._area);

      // Add circle to connect and interact with the control points
      const circle = svg.selectAll("circle").data(this._controlPoints)

      circle.enter()
          .append("circle")
          .attr("cx", (d) => { return this._xScale(d.x); })
          .attr("cy", (d) => { return this._yScale(d.opacity); })
          .style("fill", (d) => { return d.color; })
          .attr("r", 1e-6)
          .on("mousedown",
              (d) => {
                this._selected = this._dragged = d;
                this._last_color               = d.color;
                this._redraw();
                this._updateLockButtonState();
              })
          .on("mouseup", () => { this._mouseup(); })
          .on("contextmenu",
              (d, i) => {
                // react on right-clicking
                d3.event.preventDefault();
                d.color  = this.colorPicker.value;
                d.locked = true;
                this._redraw();
                this._updateLockButtonState();
              })
          .transition()
          .duration(750)
          .attr("r", (d) => { return d.locked ? 6.0 : 4.0; });

      circle.classed("selected", (d) => { return d === this._selected; })
          .style("fill", (d) => { return d.color; })
          .attr("cx", (d) => { return this._xScale(d.x); })
          .attr("cy", (d) => { return this._yScale(d.opacity); })
          .attr("r", (d) => { return d.locked ? 6.0 : 4.0; });

      circle.exit().remove();

      // Create a linear gradient definition of the control points
      const gradient = svg.select("linearGradient").selectAll("stop").data(this._controlPoints);

      gradient.enter()
          .append("stop")
          .attr("stop-color", (d) => { return d.color; })
          .attr("stop-opacity", (d) => { return d.opacity; })
          .attr("offset", (d) => {
            const l =
                (this._controlPoints[this._controlPoints.length - 1].x - this._controlPoints[0].x);
            return "" + ((d.x - this._controlPoints[0].x) / l * 100) + "%";
          });

      gradient.attr("stop-color", (d) => { return d.color; })
          .attr("stop-opacity", (d) => { return d.opacity; })
          .attr("offset", (d) => {
            const l =
                (this._controlPoints[this._controlPoints.length - 1].x - this._controlPoints[0].x);
            return "" + ((d.x - this._controlPoints[0].x) / l * 100) + "%";
          });

      gradient.exit().remove();

      const callback = CosmoScout.callbacks.find(this.callback);
      if (callback !== undefined) {
        callback(this.getJsonString());
      }

      if (d3.event) {
        d3.event.preventDefault();
        d3.event.stopPropagation();
      }
    }

    /////// User interaction related event callbacks ////////

    _mousedown() {
      const pos   = d3.mouse(this._svg.node());
      const point = {
        "x": this._xScale.invert(Math.max(0, Math.min(pos[0] - this._margin.left, this._width))),
        "opacity":
            this._yScale.invert(Math.max(0, Math.min(pos[1] - this._margin.top, this._height)))
      };
      this._selected = this._dragged = point;
      const bisect                   = d3.bisector((a, b) => { return a.x - b.x; }).left;
      const indexPos                 = bisect(this._controlPoints, point);
      this._controlPoints.splice(indexPos, 0, point);
      this._redraw();
      this._updateLockButtonState();
    }

    _mousemove() {
      if (!this._dragged)
        return;

      function equalPoint(a, index, array) {
        return a.x == this.x && a.opacity == this.opacity && a.color == this.color;
      };
      const index = this._controlPoints.findIndex(equalPoint, this._selected);
      if (index == -1)
        return;
      const m        = d3.mouse(this._svg.node());
      this._selected = this._dragged = this._controlPoints[index];
      if (index != 0 && index != this._controlPoints.length - 1) {
        this._dragged.x =
            this._xScale.invert(Math.max(0, Math.min(this._width, m[0] - this._margin.left)));
      }
      this._dragged.opacity =
          this._yScale.invert(Math.max(0, Math.min(this._height, m[1] - this._margin.top)));
      const bisect        = d3.bisector((a, b) => { return a.x - b.x; }).left;
      const bisect2       = d3.bisector((a, b) => { return a.x - b.x; }).right;
      const virtualIndex  = bisect(this._controlPoints, this._dragged);
      const virtualIndex2 = bisect2(this._controlPoints, this._dragged);
      if (virtualIndex < index) {
        this._controlPoints.splice(virtualIndex, 1);
      } else if (virtualIndex > index) {
        this._controlPoints.splice(index + 1, 1);
      } else if (virtualIndex2 - index >= 2) {
        this._controlPoints.splice(index + 1, 1);
      }
      this._redraw();
    }

    _mouseup() {
      if (!this._dragged)
        return;
      this._dragged = null;
    }

    getJsonString() {
      // Utility functions for parsing colors
      function colorHexToComponents(hexString) {
        const red   = parseInt(hexString.substring(1, 3), 16) / 255.0;
        const green = parseInt(hexString.substring(3, 5), 16) / 255.0;
        const blue  = parseInt(hexString.substring(5, 7), 16) / 255.0;
        return [red, green, blue];
      }

      function colorRgbToComponents(rgbString) {
        return rgbString.substring(4, rgbString.length - 1).split(",").map(s => s.trim() / 255.0);
      }

      function colorToComponents(colorString) {
        if (colorString.startsWith("#")) {
          return colorHexToComponents(colorString);
        } else if (colorString.startsWith("rgb")) {
          return colorRgbToComponents(colorString);
        }
      }

      const exportObject = {};
      exportObject.RGB   = {};
      exportObject.Alpha = {};

      const min   = this._xScale.domain()[0];
      const max   = this._xScale.domain()[1];
      const range = max - min;
      this._controlPoints.forEach((controlPoint) => {
        const position = controlPoint.x;
        const opacity  = controlPoint.opacity;
        if (controlPoint.locked) {
          exportObject.RGB[(position - min) / range] = colorToComponents(controlPoint.color);
        }
        exportObject.Alpha[(position - min) / range] = opacity;
      });

      return JSON.stringify(exportObject);
    }

    loadTransferFunction(jsonTransferFunction) {
      let transferFunction = JSON.parse(jsonTransferFunction);
      const points         = [];
      if (Array.isArray(transferFunction)) {
        // Paraview transfer function format
        transferFunction = transferFunction[0];
        let index        = 0;
        for (let i = 0; i < transferFunction.RGBPoints.length;) {
          points[index]          = {};
          points[index].position = transferFunction.RGBPoints[i++];
          points[index].r        = transferFunction.RGBPoints[i++];
          points[index].g        = transferFunction.RGBPoints[i++];
          points[index].b        = transferFunction.RGBPoints[i++];
          index++;
        }
        if (transferFunction.Points != null) {
          for (let i = 0; i < transferFunction.Points.length;) {
            if (!points.some(
                    point => Number(point.position) === Number(transferFunction.Points[i]))) {
              points[index]          = {};
              points[index].position = transferFunction.Points[i++];
              points[index].a        = transferFunction.Points[i++];
              i += 2;
              index++;
            } else {
              points.find(point => Number(point.position) === Number(transferFunction.Points[i]))
                  .a = transferFunction.Points[++i];
              i += 3;
            }
          }
        } else {
          points[0].a                 = 0;
          points[points.length - 1].a = 1;
        }
      } else {
        // Cosmoscout transfer function format
        let index = 0;
        Object.keys(transferFunction.RGB).forEach((position) => {
          points[index]          = {};
          points[index].position = position;
          points[index].r        = transferFunction.RGB[position][0];
          points[index].g        = transferFunction.RGB[position][1];
          points[index].b        = transferFunction.RGB[position][2];
          ++index;
        });
        Object.keys(transferFunction.Alpha).forEach((position) => {
          if (!points.some(point => Number(point.position) === Number(position))) {
            points[index]          = {};
            points[index].position = position;
            points[index].a        = transferFunction.Alpha[position];
            ++index;
          } else {
            points.find(point => Number(point.position) === Number(position)).a =
                transferFunction.Alpha[position];
          }
        });
      }

      points.sort((a, b) => { return a.position - b.position; });
      const min = points[0].position;
      const max = points[points.length - 1].position;
      points.forEach((point, index) => {
        if (typeof point.a === 'undefined') {
          var right = points.slice(index, points.length).find(point => point.hasOwnProperty("a"));
          var left  = points.slice(0, index).reverse().find(point => point.hasOwnProperty("a"));
          if (left && right) {
            point.a = d3.interpolateNumber(left.a, right.a)(
                (point.position - left.position) / (right.position - left.position));
          } else if (left) {
            point.a = left.a;
          } else if (right) {
            point.a = right.a;
          }
        }
      });
      this._updateControlPoints(points.map((point) => {
        return {
          x: (point.position - min) * (255.0 / (max - min)),
          opacity: point.a,
          color: (typeof point.r !== 'undefined')
                     ? "#" + (0x1000000 +
                                 (point.b * 255 | (point.g * 255 << 8) | (point.r * 255 << 16)))
                                 .toString(16)
                                 .slice(1)
                     : "#000000",
          locked: (typeof point.r !== 'undefined')
        };
      }));
      this._redraw();
    }

    setAvailableTransferFunctions(availableFiles) {
      let options = "";
      availableFiles.forEach((file) => { options += `<option>${file}</option>`; });
      const importSelect = $("#transferFunctionEditor\\.importSelect-" + this.id);
      importSelect.html(options);
      importSelect.selectpicker();
      importSelect.selectpicker("refresh");
    }
  }

  /**
   * Transfer function editor API
   */
  class TransferFunctionEditorApi extends IApi {
    /**
     * @inheritDoc
     */
    name = "transferFunctionEditor";

    /**
     * id suffix of the next created transfer function editor
     */
    _nextId = 0;

    _editors = [];

    _availableFiles = [];

    /**
     * @inheritDoc
     */
    init() {
      const templateHtml        = `
        <div>
          <div class="row">
            <div class="col-12">
              <svg id="transferFunctionEditor.graph-%ID%"></svg>
            </div>
          </div>
          <div class="row">
            <div class="col-2">
              <a id="transferFunctionEditor.colorLock-%ID%" class="btn glass block" title="Lock Color">
                <i class="material-icons">lock</i>
              </a>
            </div>
            <div class="col-3">
              <input type="text" class="form-control color-input" id="transferFunctionEditor.colorPicker-%ID%" style="color: black;" value="#FF0000">
            </div>
          </div>
          <div class="row">
            <div class="col-5">
              <button id="transferFunctionEditor.export-%ID%" class="waves-effect waves-light block btn glass text">Export</button>
            </div>
            <div class="col-7">
              <input type="text" id="transferFunctionEditor.exportLocation-%ID%" placeholder="Filename" class="text-input text">
            </div>
          </div>
          <div class="row">
            <div class="col-5">
              <button id="transferFunctionEditor.import-%ID%" class="waves-effect waves-light block btn glass text">Import</button>
            </div>
            <div class="col-7">
              <select id="transferFunctionEditor.importSelect-%ID%">
                <option value="-1">none</option>
              </select>
            </div>
          </div>
        </div>
      `;
      const templateElement     = document.createElement("template");
      templateElement.id        = "transferFunctionEditor-template";
      templateElement.innerHTML = templateHtml;
      document.body.appendChild(templateElement);
    }

    create(containerId, callback, options) {
      const editor = CosmoScout.gui.loadTemplateContent("transferFunctionEditor");
      if (editor === false) {
        console.warn('"#transferFunctionEditor-template" could not be loaded!');
        return;
      }

      document.getElementById(containerId).innerHTML =
          editor.innerHTML.replace(/%ID%/g, this._nextId);

      const transferFunctionEditor = new TransferFunctionEditor(this._nextId, callback, options);
      transferFunctionEditor.setAvailableTransferFunctions(this._availableFiles);
      this._editors.push(transferFunctionEditor);
      this._nextId++;
      return transferFunctionEditor;
    }

    setAvailableTransferFunctions(availableFilesJson) {
      this._availableFiles = JSON.parse(availableFilesJson);
      this._editors.forEach(
          (editor) => { editor.setAvailableTransferFunctions(this._availableFiles); });
    }

    loadTransferFunction(jsonTransferFunction, editorId) {
      this._editors.find(e => e.id == editorId).loadTransferFunction(jsonTransferFunction);
    }
  }

  CosmoScout.init(TransferFunctionEditorApi);
})();
