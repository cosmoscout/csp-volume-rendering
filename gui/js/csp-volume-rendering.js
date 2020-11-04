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
            CosmoScout.gui.initSliderRange("volumeRendering.setSamplingRate", { "min": 0.001, "33%": 0.01, "66%": 0.1, "max": 1 }, 0.001, [0.005]);
            CosmoScout.gui.initSlider("volumeRendering.setSunStrength", 0, 10, 0.1, [1]);
            CosmoScout.gui.initSlider("volumeRendering.setDensityScale", 0, 10, 0.1, [1]);

            // Trigger "setTimestep" callback on "update" event
            var timestepSlider = document.querySelector(`[data-callback="volumeRendering.setTimestep"]`);
            timestepSlider.dataset.event = "update";

            this.createElements();
            this.ready();
        }

        play() {
            const playButtonIcon = $("#volumeRendering\\.play i");
            if (this.playing) {
                clearInterval(this.playHandle);
                playButtonIcon.html("play_arrow");
                this.playing = false;
            }
            else {
                const timeSlider = document.querySelector(`[data-callback="volumeRendering.setTimestep"]`).noUiSlider;
                this.time = parseInt(timeSlider.get());
                let prevNext = this.time;
                this.playHandle = setInterval(() => {
                    // reverse() changes the original array, so it is called twice to return the array to the original state
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
                    const speedSlider = document.querySelector(`[data-callback="volumeRendering.setAnimationSpeed"]`).noUiSlider;
                    this.time += parseInt(speedSlider.get()) / 10;
                }, 100);
                playButtonIcon.html("pause");
                this.playing = true;
            }
        }

        setTimesteps(timestepsJson) {
            this.timesteps = JSON.parse(timestepsJson);
            this.timesteps.sort();
            var min = this.timesteps[0];
            var max = this.timesteps[this.timesteps.length - 1];
            var range = {};
            this.timesteps.forEach((t, i) => {
                var percent = ((t - min) / (max - min) * 100) + "%";
                if (t == min) {
                    percent = "min";
                }
                else if (t == max) {
                    percent = "max";
                }
                range[percent] = [];
                range[percent][0] = t;
                if (t != max) {
                    range[percent][1] = this.timesteps[i + 1] - t;
                }
            });
            CosmoScout.gui.initSliderRange("volumeRendering.setTimestep", range, this.timesteps[0], [100]);
        }

        ready() {
            // Access the svg dom element
            this.svg = d3.select("#volumeRendering\\.tfGraph");
            this._width = +this.svg.attr("width") - this.margin.left - this.margin.right;
            this._height = +this.svg.attr("height") - this.margin.top - this.margin.bottom - 15;
            this.initialized = true;
            this.initializeElements();
            this.drawChart();
        }

        createElements() {
            // Custom margins
            this.margin = {
                top: 10,
                right: 20,
                bottom: 20,
                left: 25
            };
            this.formatCount = d3.format(",.0f");

            // Axis scales
            this.xScale = d3.scale.linear();
            this.yScale = d3.scale.linear();

            // Area for the opacity map representation
            this.area = d3.svg.area();

            // Keep track of control points interaction
            this.dragged = null;
            this.selected = null;
            this.last_color = 'green';

            this.controlPoints = [];
        }

        initializeElements() {
            var extent = [0, 255];
            if (this.fitToData && this._data && this._data.length > 0) {
                extent = d3.extent(this._data);
            }
            var me = this;
            this.xScale
                .rangeRound([0, this._width])
                .domain(extent);
            this.yScale
                .domain([0, 1])
                .range([this._height, 0]);
            if (this.controlPoints.length == 0) {
                this.controlPoints.push({
                    'x': extent[0],
                    'opacity': 0,
                    'color': '#0000FF',
                    'locked': true
                });
                this.controlPoints.push({
                    'x': extent[1],
                    'opacity': 1,
                    'color': '#FF0000',
                    'locked': true
                });
            }
            this.selected = this.controlPoints[1];
            this.area
                .x(function (d) {
                    return me.xScale(d.x);
                })
                .y0(function (d) {
                    return me.yScale(d.opacity);
                })
                .y1(this._height);

            // Access the color selector
            this.colorPicker = $("#volumeRendering\\.tfColorPicker").get(0);
            this.colorPicker.picker.on("change", () => {
                me.selected.color = this.colorPicker.value;
                me.redraw();
            });
            // Export button listener
            $("#volumeRendering\\.tfExport").on("click", function () {
                CosmoScout.callbacks.volumeRendering.exportTransferFunction($("#volumeRendering\\.tfExportLocation").val(), me.getJsonString());
            });
            // Import button listener
            $("#volumeRendering\\.tfImport").on("click", function () {
                CosmoScout.callbacks.volumeRendering.importTransferFunction($("#volumeRendering\\.tfImportSelect").val());
            });

            // Lock button listener
            $("#volumeRendering\\.tfColorLock").on("click", function () {
                if (me.controlPoints.some(point => point.locked && point !== me.selected)) {
                    me.selected.locked = !me.selected.locked;
                    me.redraw();
                    me.updateLockButtonState();
                }
            });
        }

        updateLockButtonState() {
            var colorLockButton = $("#volumeRendering\\.tfColorLock i");
            if (this.selected.locked) {
                $(colorLockButton).html("lock");
            }
            else {
                $(colorLockButton).html("lock_open");
            }
        }

        // Perform the drawing
        drawChart() {
            var me = this;
            var g = this.svg.append("g")
                .attr("transform", "translate(" + this.margin.left + "," + this.margin.top + ")")
                .on("mouseleave", function () {
                    me.mouseup();
                });

            // Gradient definitions
            g.append("defs").append("linearGradient")
                .attr("id", "volumeRendering.tfGradient")
                //.attr("gradientUnits", "userSpaceOnUse")
                .attr("gradientUnits", "objectBoundingBox")
                .attr("spreadMethod", "pad")
                .attr("x1", "0%").attr("y1", "0%")
                .attr("x2", "100%").attr("y2", "0%");
            //.attr("x1", me.xScale(0)).attr("y1", me.yScale(0))
            //.attr("x2", me.xScale(255)).attr("y2", me.yScale(0));

            // Draw control points
            g.append("path")
                .datum(me.controlPoints)
                .attr("class", "line")
                .attr("fill", "url(#volumeRendering.tfGradient)")
                .attr("stroke", "black");

            g.append("path")
                .datum(me.controlPoints)
                .attr("class", "line")
                .attr("fill", "none")
                .attr("stroke", "black")
                .call(function () {
                    me.redraw();
                });

            // Mouse interaction handler
            g.append("rect")
                .attr("y", -10)
                .attr("x", -10)
                .attr("width", me._width + 20)
                .attr("height", me._height + 20)
                .style("opacity", 0)
                .on("mousedown", function () {
                    me.mousedown();
                })
                .on("mouseup", function () {
                    me.mouseup();
                })
                .on("mousemove", function () {
                    me.mousemove();
                });

            g.append("text")
                .attr("transform", "translate(" + (me._width + me.margin.left - 5) + " ," + (me._height + me.margin.top + 20) + ")")
                .attr("class", "label")
                .text("Unit");

            // Draw axis
            var xTicks = me.xScale.ticks(me.numberTicks);
            xTicks[xTicks.length - 1] = me.xScale.domain()[1];
            g.append("g")
                .attr("class", "axis axis--x")
                .attr("transform", "translate(0," + me._height + ")")
                .call(d3.svg.axis().scale(me.xScale).orient("bottom").tickValues(xTicks));

            g.append("g")
                .attr("class", "axis axis--y")
                .attr("transform", "translate(0, 0)")
                .call(d3.svg.axis().scale(me.yScale).orient("left").ticks(me.numberTicks));
        }

        // update scales with new data input
        updateScales() {
            if (this.fitToData) {
                var dataExtent = [];
                if (this._data && this._data.length > 0) {
                    dataExtent = d3.extent(this._data);
                }
                if (dataExtent[0] == dataExtent[1]) {
                    dataExtent[1] += 1;
                }
                this.xScale.domain(dataExtent);
            } else {
                this.xScale.domain([0, 255]);
            }
        }

        // update the axis with the new data input
        updateAxis() {
            let svg = d3.select("svg").select("g");
            var xTicks = this.xScale.ticks(this.numberTicks);
            xTicks[xTicks.length - 1] = this.xScale.domain()[1];
            svg.selectAll(".axis.axis--x").call(d3.axisBottom(this.xScale).tickValues(xTicks));
        }

        updateControlPoints(controlPoints) {
            var updateScale = d3.scale.linear()
                .domain(d3.extent(controlPoints, point => point.x))
                .range(this.xScale.domain());
            controlPoints.forEach(point => point.x = updateScale(point.x));
            this.controlPoints = controlPoints;
        }

        // Update the chart content
        redraw() {
            var me = this;

            if (!me.controlPoints.some(point => point.locked)) {
                me.selected.locked = !me.selected.locked;
                me.redraw();
                me.updateLockButtonState();
            }
            me.controlPoints.forEach((point, index) => {
                if (index == 0) {
                    point.x = me.xScale.invert(0);
                }
                else if (index == me.controlPoints.length - 1) {
                    point.x = me.xScale.invert(me._width);
                }

                if (!point.locked) {
                    var right = me.controlPoints.slice(index, me.controlPoints.length).find(point => point.locked);
                    var left = me.controlPoints.slice(0, index).reverse().find(point => point.locked);
                    if (left && right) {
                        point.color = d3.interpolateRgb(left.color, right.color)((point.x - left.x) / (right.x - left.x));
                    }
                    else if (left) {
                        point.color = left.color;
                    }
                    else if (right) {
                        point.color = right.color;
                    }
                }
            });

            if (CosmoScout.callbacks.volumeRendering != null && CosmoScout.callbacks.volumeRendering.setTransferFunction != null)
                CosmoScout.callbacks.volumeRendering.setTransferFunction(this.getJsonString());

            var svg = d3.select("#volumeRendering\\.tfGraph").select("g");
            svg.selectAll("path.line").datum(me.controlPoints).attr("d", me.area);

            // Add circle to connect and interact with the control points
            var circle = svg.selectAll("circle").data(me.controlPoints)

            circle.enter().append("circle")
                .attr("cx", function (d) {
                    return me.xScale(d.x);
                })
                .attr("cy", function (d) {
                    return me.yScale(d.opacity);
                })
                .style("fill", function (d) {
                    return d.color;
                })
                .attr("r", 1e-6)
                .on("mousedown", function (d) {
                    me.selected = me.dragged = d;
                    me.last_color = d.color;
                    me.redraw();
                    me.updateLockButtonState();
                })
                .on("mouseup", function () {
                    me.mouseup();
                })
                .on("contextmenu", function (d, i) {
                    // react on right-clicking
                    d3.event.preventDefault();
                    d.color = this.colorPicker.value;
                    d.locked = true;
                    me.redraw();
                    me.updateLockButtonState();
                })
                .transition()
                .duration(750)
                .attr("r", function (d) {
                    return d.locked ? 6.0 : 4.0;
                });

            circle.classed("selected", function (d) {
                return d === me.selected;
            })
                .style("fill", function (d) {
                    return d.color;
                })
                .attr("cx", function (d) {
                    return me.xScale(d.x);
                })
                .attr("cy", function (d) {
                    return me.yScale(d.opacity);
                })
                .attr("r", function (d) {
                    return d.locked ? 6.0 : 4.0;
                });

            circle.exit().remove();

            // Create a linear gradient definition of the control points
            var gradient = svg.select("linearGradient").selectAll("stop").data(me.controlPoints);

            gradient.enter().append("stop")
                .attr("stop-color", function (d) {
                    return d.color;
                })
                .attr("stop-opacity", function (d) {
                    return d.opacity;
                })
                .attr("offset", function (d) {
                    var l = (me.controlPoints[me.controlPoints.length - 1].x - me.controlPoints[0].x);
                    return "" + ((d.x - me.controlPoints[0].x) / l * 100) + "%";
                });

            gradient.attr("stop-color", function (d) {
                return d.color;
            })
                .attr("stop-opacity", function (d) {
                    return d.opacity;
                })
                .attr("offset", function (d) {
                    var l = (me.controlPoints[me.controlPoints.length - 1].x - me.controlPoints[0].x);
                    return "" + ((d.x - me.controlPoints[0].x) / l * 100) + "%";
                });

            gradient.exit().remove();

            if (d3.event) {
                d3.event.preventDefault();
                d3.event.stopPropagation();
            }

            // Override dirty checking
            var controlPoints = this.controlPoints;
            this.controlPoints = [];
            this.controlPoints = controlPoints;
        }

        /**
         * Update x axis label
         */
        updateUnit(unit) {
            if (unit == "")
                unit = "-";
            var me = this;
            var svg = d3.select("svg").select("g");
            svg.select(".label").text("Unit: " + unit);
        }

        /////// User interaction related event callbacks ////////

        mousedown() {
            var me = this;
            var pos = d3.mouse(me.svg.node());
            var point = {
                "x": me.xScale.invert(Math.max(0, Math.min(pos[0] - me.margin.left, me._width))),
                "opacity": me.yScale.invert(Math.max(0, Math.min(pos[1] - me.margin.top, me._height)))
            };
            me.selected = me.dragged = point;
            var bisect = d3.bisector(function (a, b) {
                return a.x - b.x;
            }).left;
            var indexPos = bisect(me.controlPoints, point);
            me.controlPoints.splice(indexPos, 0, point);
            me.redraw();
            me.updateLockButtonState();
        }

        mousemove() {
            if (!this.dragged) return;

            function equalPoint(a, index, array) {
                return a.x == this.x && a.opacity == this.opacity && a.color == this.color;
            };
            var index = this.controlPoints.findIndex(equalPoint, this.selected);
            if (index == -1) return;
            var m = d3.mouse(this.svg.node());
            this.selected = this.dragged = this.controlPoints[index];
            if (index != 0 && index != this.controlPoints.length - 1) {
                this.dragged.x = this.xScale.invert(Math.max(0, Math.min(this._width, m[0] - this.margin.left)));
            }
            this.dragged.opacity = this.yScale.invert(Math.max(0, Math.min(this._height, m[1] - this.margin.top)));
            var bisect = d3.bisector(function (a, b) {
                return a.x - b.x;
            }).left;
            var bisect2 = d3.bisector(function (a, b) {
                return a.x - b.x;
            }).right;
            var virtualIndex = bisect(this.controlPoints, this.dragged);
            var virtualIndex2 = bisect2(this.controlPoints, this.dragged);
            if (virtualIndex < index) {
                this.controlPoints.splice(virtualIndex, 1);
            } else if (virtualIndex > index) {
                this.controlPoints.splice(index + 1, 1);
            } else if (virtualIndex2 - index >= 2) {
                this.controlPoints.splice(index + 1, 1);
            }
            this.redraw();
        }

        mouseup() {
            if (!this.dragged) return;
            this.dragged = null;
        }

        getJsonString() {
            // Utility functions for parsing colors
            function colorHexToComponents(hexString) {
                const red = parseInt(hexString.substring(1, 3), 16) / 255.0;
                const green = parseInt(hexString.substring(3, 5), 16) / 255.0;
                const blue = parseInt(hexString.substring(5, 7), 16) / 255.0;
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
            exportObject.RGB = {};
            exportObject.Alpha = {};

            const min = this.xScale.domain()[0];
            const max = this.xScale.domain()[1];
            const range = max - min;
            this.controlPoints.forEach((controlPoint) => {
                const position = controlPoint.x;
                const opacity = controlPoint.opacity;
                if (controlPoint.locked) {
                    exportObject.RGB[(position - min) / range] = colorToComponents(controlPoint.color);
                }
                exportObject.Alpha[(position - min) / range] = opacity;
            });

            return JSON.stringify(exportObject);
        }

        loadTransferFunction(jsonTransferFunction) {
            let transferFunction = JSON.parse(jsonTransferFunction);
            const points = [];
            if (Array.isArray(transferFunction)) {
                // Paraview transfer function format
                transferFunction = transferFunction[0];
                let index = 0;
                for (let i = 0; i < transferFunction.RGBPoints.length;) {
                    points[index] = {};
                    points[index].position = transferFunction.RGBPoints[i++];
                    points[index].r = transferFunction.RGBPoints[i++];
                    points[index].g = transferFunction.RGBPoints[i++];
                    points[index].b = transferFunction.RGBPoints[i++];
                    index++;
                }
                if (transferFunction.Points != null) {
                    for (let i = 0; i < transferFunction.Points.length;) {
                        if (!points.some(point => Number(point.position) === Number(transferFunction.Points[i]))) {
                            points[index] = {};
                            points[index].position = transferFunction.Points[i++];
                            points[index].a = transferFunction.Points[i++];
                            i += 2;
                            index++;
                        } else {
                            points.find(point => Number(point.position) === Number(transferFunction.Points[i])).a = transferFunction.Points[++i];
                            i += 3;
                        }
                    }
                }
                else {
                    points[0].a = 0;
                    points[points.length - 1].a = 1;
                }
            } else {
                // Cosmoscout transfer function format
                let index = 0;
                Object.keys(transferFunction.RGB).forEach((position) => {
                    points[index] = {}
                    points[index].position = position;
                    points[index].r = transferFunction.RGB[position][0];
                    points[index].g = transferFunction.RGB[position][1];
                    points[index].b = transferFunction.RGB[position][2];
                    ++index;
                });
                Object.keys(transferFunction.Alpha).forEach((position) => {
                    if (!points.some(point => Number(point.position) === Number(position))) {
                        points[index] = {};
                        points[index].position = position;
                        points[index].a = transferFunction.Alpha[position];
                        ++index;
                    } else {
                        points.find(point => Number(point.position) === Number(position)).a = transferFunction.Alpha[position];
                    }
                });
            }

            points.sort((a, b) => {
                return a.position - b.position;
            });
            let min = points[0].position;
            let max = points[points.length - 1].position;
            points.forEach((point, index) => {
                if (typeof point.a === 'undefined') {
                    var right = points.slice(index, points.length).find(point => point.hasOwnProperty("a"));
                    var left = points.slice(0, index).reverse().find(point => point.hasOwnProperty("a"));
                    if (left && right) {
                        point.a = d3.interpolateNumber(left.a, right.a)((point.position - left.position) / (right.position - left.position));
                    } else if (left) {
                        point.a = left.a;
                    } else if (right) {
                        point.a = right.a;
                    }
                }
            });
            this.updateControlPoints(points.map((point) => {
                return {
                    x: (point.position - min) * (255.0 / (max - min)),
                    opacity: point.a,
                    color: (typeof point.r !== 'undefined') ? "#" + (0x1000000 + (point.b * 255 | (point.g * 255 << 8) | (point.r * 255 << 16))).toString(16).slice(1) : "#000000",
                    locked: (typeof point.r !== 'undefined')
                };
            }));
            this.redraw();
        }

        setAvailableTransferFunctions(availableFilesJson) {
            let options = "";
            const availableFiles = JSON.parse(availableFilesJson);
            availableFiles.forEach((file) => {
                options += `<option>${file}</option>`;
            });
            const importSelect = $("#volumeRendering\\.tfImportSelect");
            importSelect.html(options);
            importSelect.selectpicker();
            importSelect.selectpicker("refresh");
        }
    }

    CosmoScout.init(VolumeRenderingApi);
})();