<template>
    <div class="nurie-view">
        <button class="nurie-view_load-button" @click="onClick"> {{ !generaterModel ? 'モデル読み込み': 'モデル読み込み済み' }} </button>
        <button class="nurie-view_load-button" @click="convertImage"> 変換 </button>
        <button class="nurie-view_load-button" @click="onClickLoadModel"> ポケモン書き込み </button>
        <div class="nurie-view_edits">
            <div>
                <canvas ref="srcCanvas" class="nurie-view_canvas" width="256px" height="256px" @mousemove="canvasDraw" @mousedown="canvasDragStart" @mouseup="canvasDragEnd" @mouseout="canvasDragEnd" />
            </div>
            <div>
                <canvas ref="distCanvas" class="nurie-view_canvas" width="256px" height="256px" />
            </div>
        </div>
    </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator'
import * as tf from '@tensorflow/tfjs'

@Component
export default class NurieView extends Vue {

    @Prop() modelPath!: string
    
    // 生成モデル
    generaterModel: tf.LayersModel | null = null

    canvas: HTMLCanvasElement | null = null
    ctx: CanvasRenderingContext2D | null = null
    isDrag: boolean = false

    outCanvas: HTMLCanvasElement | null = null
    outCtx: CanvasRenderingContext2D | null = null


    mounted() {
        this.canvasInit()
        this.outCanvasInit()
    }

    onClick() {
        if ( !this.generaterModel ) {
            this.loadModel(this.modelPath)
        }
    }

    onClickLoadModel() {
        this.loadImage('/examples/edges/001.png')
    }
    
    async loadModel(modelPath: string) {
        try {
            this.generaterModel = await tf.loadLayersModel(modelPath)
        } catch ( e ) {
            // 読み込めなかった処理
        }
    }

    canvasInit() {
        this.canvas = <HTMLCanvasElement>this.$refs.srcCanvas
        this.ctx    = this.canvas.getContext('2d')
        if ( !!this.ctx ) {
            this.ctx.fillStyle = 'white'
            this.ctx.fillRect(0, 0, 256, 256)
        }
    }

    outCanvasInit() {
        this.outCanvas = <HTMLCanvasElement>this.$refs.distCanvas
        this.outCtx    = this.outCanvas.getContext('2d')
    }

    canvasDraw(e: MouseEvent) {
        const x = e.offsetX
        const y = e.offsetY
        if ( !this.isDrag ) { return }
        if ( !!this.ctx ) {
            this.ctx.lineTo(x, y)
            this.ctx.stroke()
        }
    }

    canvasDragStart(e: MouseEvent) {
        const x = e.offsetX
        const y = e.offsetY
        if ( !!this.ctx ) {
            this.ctx.beginPath()
            this.ctx.lineTo(x, y)
            this.ctx.stroke()
        }
        this.isDrag = true
    }

    canvasDragEnd(e: MouseEvent) {
        if ( !!this.ctx ) this.ctx.closePath()
        this.isDrag = false
    }

    loadImage(imgSrc: string) {
        const img = new Image()
        img.src   = imgSrc
        img.onload = () => {
            if ( !!this.ctx ) this.ctx.drawImage(img, 0, 0)
        }
    }

    convertImage() {
        if ( !!this.canvas && !!this.ctx && !!this.generaterModel ) {
            const imageData = this.ctx.getImageData(0, 0, 256, 256)
            const edge      = tf.browser.fromPixels(imageData, 1).reshape([1, 256, 256, 1]).div(255)
            const out       = (<tf.Tensor>this.generaterModel.predict(edge)).reshape([256, 256, 3]).mul(255)
            const outImage  = this.toPixels(<tf.Tensor3D>out)
            if ( !!this.outCtx ) this.outCtx.putImageData(outImage, 0, 0)
        }
    }

    toPixels(tensor: tf.Tensor3D) {
        const pixels = tensor.dataSync()
        const imageData = new ImageData(tensor.shape[0], tensor.shape[1])
        for (let i = 0; i < pixels.length/3; i++) {
            imageData.data[i*4+0] = pixels[i*3+0]
            imageData.data[i*4+1] = pixels[i*3+1]
            imageData.data[i*4+2] = pixels[i*3+2]
            imageData.data[i*4+3] = 255
        }
        return imageData
    }

}
</script>

<style lang="scss">
.nurie-view {
    background: red;
    height: 80vh;
    &_load-button {
        margin-top: 2vh;
        width: 20%;
        height: 5vh;
    }
    &_edits {
        height: 60vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        flex-wrap: wrap;
    }
    &_canvas {
        background: white;
        width: 256px;
        height: 256px;
    }
}
</style>