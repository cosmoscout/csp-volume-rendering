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

    call(desc) {
      console.log('Starting call');
      this.startTime      = window.performance.now();
      const configuration = {iceServers: [{urls: ["stun:turn2.l.google.com"]}]};
      this.pc2            = new RTCPeerConnection(configuration);
      this.pc2.addEventListener('icecandidate', e => this.onIceCandidate(e));
      this.pc2.addEventListener('track', e => this.gotRemoteStream(e));
      this.onCreateOfferSuccess(desc);
    }

    onCreateOfferSuccess(desc) {
      this.pc2.setRemoteDescription(desc);
      console.log('pc2 createAnswer start');
      // Since the 'remote' side has no media stream we need
      // to pass in the right constraints in order for it to
      // accept the incoming offer of audio and video.
      try {
        this.pc2.createAnswer()
            .then(answer => this.onCreateAnswerSuccess(answer))
            .catch(e => console.log("Error: " + e));
      } catch (e) { console.log("Error: " + e); }
    }

    gotRemoteStream(e) {
      if (this.remoteVideo.srcObject !== e.streams[0]) {
        this.remoteVideo.srcObject = e.streams[0];
        console.log('pc2 received remote stream');
      }
    }

    onCreateAnswerSuccess(desc) {
      try {
        this.pc2.setLocalDescription(desc);
        console.log("Answer: " + JSON.stringify(desc));
      } catch (e) { console.log("Error: " + e); }
    }

    onIceCandidate(event) {
      try {
        console.log("ICE: " + JSON.stringify(event.candidate));
      } catch (e) { console.log("Error: " + e); }
    }

    capture() {
      const canvas  = document.createElement("canvas");
      canvas.width  = this.remoteVideo.videoWidth;
      canvas.height = this.remoteVideo.videoHeight;
      canvas.getContext("2d").drawImage(
          this.remoteVideo, 0, 0, this.remoteVideo.videoWidth, this.remoteVideo.videoHeight);
      const data = canvas.toDataURL("image/png");
      document.getElementById("captureImg").setAttribute("src", data);
      CosmoScout.callbacks.volumeRendering.captureColorImage(
          data.replace("data:image/png;base64,", ""));
    }

    /**
     * @inheritDoc
     */
    init() {
      this.remoteVideo = document.getElementById("remotevideo");

      this.remoteVideo.addEventListener('resize', () => {
        console.log(`Remote video size changed to ${this.remoteVideo.videoWidth}x${
            this.remoteVideo.videoHeight}`);
        // We'll use the first onsize callback as an indication that video has started
        // playing out.
        if (this.startTime) {
          const elapsedTime = window.performance.now() - this.startTime;
          console.log('Setup time: ' + elapsedTime.toFixed(3) + 'ms');
          this.startTime = null;
        }
      });

      CosmoScout.gui.initInputs();
      CosmoScout.gui.initSlider("volumeRendering.setAnimationSpeed", 10, 1000, 10, [100]);
      CosmoScout.gui.initSlider("volumeRendering.setResolution", 32, 2048, 32, [256]);
      CosmoScout.gui.initSliderRange("volumeRendering.setSamplingRate",
          {"min": 0.001, "33%": 0.01, "66%": 0.1, "max": 1}, [0.005]);
      CosmoScout.gui.initSlider("volumeRendering.setSunStrength", 0, 10, 0.1, [1]);
      CosmoScout.gui.initSlider("volumeRendering.setDensityScale", 0, 10, 0.1, [1]);

      // Trigger "setTimestep" callback on "update" event
      var timestepSlider = document.querySelector(`[data-callback="volumeRendering.setTimestep"]`);
      timestepSlider.dataset.event = "update";

      this.progressBar = document.getElementById("volumeRendering.progressBar");

      this.transferFunctionEditor = CosmoScout.transferFunctionEditor.create(
          document.getElementById("volumeRendering.tfEditor"), this.setTransferFunction);
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

    loadTransferFunction(path) {
      CosmoScout.callbacks.transferFunctionEditor.importTransferFunction(
          path, this.transferFunctionEditor.id);
    }
  }

  CosmoScout.init(VolumeRenderingApi);
})();
