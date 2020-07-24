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
            CosmoScout.gui.initSlider("volumeRendering.setResolution", 32, 2048, 32, [256]);
            CosmoScout.gui.initSliderRange("volumeRendering.setSamplingRate", {"min": 0.001, "33%": 0.01, "66%": 0.1, "max": 1}, 0.001, [0.005]);
        }
    }

    CosmoScout.init(VolumeRenderingApi);
})();