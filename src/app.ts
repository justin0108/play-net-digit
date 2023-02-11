/* eslint-disable @typescript-eslint/no-unused-vars */
/* eslint-disable @typescript-eslint/no-non-null-assertion */
import { TRAINED_NET_CUSTOM, TOTAL_IMAGES, TRAINED_NET_BRAINJS } from './constants';
import { MnistReader } from './MnistReader';
import { Panel } from './Panel';
import { NeuralNet } from './NeuralNet';
import { MnistEntry } from './MnistEntry';
import { randomNumber } from './helper';
import { Chart } from 'chart.js';
import { NeuralNetBrainJS } from './NeuralNetBrainJS';

/**
 * setup the event listener
 */
const NET = document.getElementById('NET') as HTMLInputElement;
const canvasDraw = document.getElementById('canvas-draw') as HTMLCanvasElement;
const canvasMnist = document.getElementById('canvas-mnist') as HTMLCanvasElement;
const canvasChart = document.getElementById('canvas-chart') as HTMLCanvasElement;
const radioCusomtNet = document.getElementById('custom-net') as HTMLInputElement;
const radioBrainjsNet = document.getElementById('brainjs-net') as HTMLInputElement;

const buttonPredict = document.getElementById('predict') as HTMLInputElement;
const buttonClear = document.getElementById('clear') as HTMLInputElement;
const buttonTrain = document.getElementById('train') as HTMLInputElement;
const fileTraining = document.getElementById('training-file') as HTMLInputElement;
const fileLabel = document.getElementById('label-file') as HTMLInputElement;

let selectedTrainingFile: File;
let selectedLabelFile: File;
let mnistEntry: MnistEntry[];
let useBrainjs = false;

const panel = new Panel(canvasDraw);
const panelMnist = new Panel(canvasMnist);
const mnistReader = new MnistReader();
const neuralNet = new NeuralNet();
const neuralNetBrainJS = new NeuralNetBrainJS();
neuralNet.restoreNetMatrix(JSON.parse(TRAINED_NET_CUSTOM));
neuralNetBrainJS.restoreNetMatrix(JSON.parse(TRAINED_NET_BRAINJS));
panel.erase();

/**
 * init the prediction chart
 */
const predictionChart = new Chart(canvasChart, {
    type: 'bar',
    data: {
        labels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        datasets: [
            {
                label: `Probability (%) [X]`,
                data: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                backgroundColor: 'lightblue',
            },
        ],
    },
    options: {
        responsive: false,
        scales: {
            yAxes: [
                {
                    ticks: {
                        beginAtZero: true,
                        callback: function(value: number, index, values) {
                            return value % 20 === 0 ? value : '';
                        },
                    },
                },
            ],
        },
    },
});

/**
 * add in all event listener
 */
radioCusomtNet.checked = true;
radioCusomtNet.addEventListener('click', _ => {
    useBrainjs = false;
});
radioBrainjsNet.addEventListener('click', _ => {
    useBrainjs = true;
});

fileTraining.addEventListener('change', async (e: Event) => {
    const target = e.target as HTMLInputElement;
    selectedTrainingFile = target.files![0];
    if (selectedTrainingFile !== undefined && selectedLabelFile !== undefined)
        mnistEntry = await mnistReader.load(selectedTrainingFile, selectedLabelFile);
});

fileLabel.addEventListener('change', async (e: Event) => {
    const target = e.target as HTMLInputElement;
    selectedLabelFile = target.files![0];
    if (selectedTrainingFile !== undefined && selectedLabelFile !== undefined)
        mnistEntry = await mnistReader.load(selectedTrainingFile, selectedLabelFile);
});

buttonPredict.addEventListener('click', _ => {
    const img = new Image();
    img.src = canvasDraw.toDataURL();
    img.onload = function() {
        const tmpCanvas = document.createElement('canvas');
        tmpCanvas.width = 28;
        tmpCanvas.height = 28;
        const tmpCtx = tmpCanvas.getContext('2d')!;
        tmpCtx.drawImage(img, 0, 0, 28, 28);
        const resizedImg = tmpCtx.getImageData(0, 0, 28, 28);
        const finalData = Array(28 * 28);
        for (let i = 0; i < finalData.length; i++) {
            finalData[i] = (255 - resizedImg.data[i * 4]) / 255;
        }
        const [predictedNumber, probabilityArray] = useBrainjs
            ? neuralNetBrainJS.predict(finalData)
            : neuralNet.predict(finalData);
        predictionChart.data.datasets![0].label = `Probability (%) [${predictedNumber}]`;
        predictionChart.data.datasets![0].data = probabilityArray;
        predictionChart.update();
    };
});

buttonClear.addEventListener('click', _ => {
    panel.erase();
});

buttonTrain.addEventListener('click', _ => {
    if (mnistEntry) {
        console.log('start training...');
        if (useBrainjs) {
            neuralNetBrainJS.train(mnistEntry);
            NET.value = JSON.stringify(neuralNetBrainJS.getNetMatrix());
        } else {
            neuralNet.train(mnistEntry);
            NET.value = JSON.stringify(neuralNet.getNetMatrix());
        }
        console.log('training ended');
    }
});

canvasMnist.addEventListener('click', async _ => {
    panelMnist.erase();
    if (mnistEntry) panelMnist.drawMnist(mnistEntry[Math.floor(randomNumber(0, TOTAL_IMAGES - 1))]);
});
