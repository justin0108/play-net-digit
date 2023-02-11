import { MnistEntry } from './MnistEntry';

// production ready net where all mathematicals details are hidden
export class NeuralNetBrainJS {
    net: any;

    constructor() {
        const config = {
            hiddenLayers: [30, 30], // array of ints for the sizes of the hidden layers in the network
            activation: 'sigmoid', // supported activation types: ['sigmoid', 'relu', 'leaky-relu', 'tanh'],
        };

        // create a simple feed forward neural network with backpropagation
        // eslint-disable-next-line no-undef
        this.net = new brain.NeuralNetwork(config);
    }

    getNetMatrix = (): any => {
        return this.net.toJSON();
    };

    restoreNetMatrix = (netMatrix: any) => {
        this.net.fromJSON(netMatrix);
    };

    predict = (input: number[]): [number, number[]] => {
        const output = this.net.run(input);
        return this.getHighestActivationNumber(output);
    };

    train = (mnistArray: MnistEntry[]) => {
        const brainJsTrainingData: any = [];
        mnistArray.forEach(item => {
            brainJsTrainingData.push({
                input: item.imageArray,
                output: this.getOutputArrayForBrainJs(item.imageLabel),
            });
        });

        this.net.train(brainJsTrainingData, {
            iterations: 20,
            log: (stats: any) => {
                console.log(stats);
            },
        });
    };

    private getOutputArrayForBrainJs = (expectedNumber: number): number[] => {
        const output = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        output[expectedNumber] = 1;
        return output;
    };

    private getHighestActivationNumber = (output: number[]): [number, number[]] => {
        let guess = 0;
        let highA = output[0];
        let totalValue = 0;
        const probabilityArray: number[] = [];
        for (let i = 0; i < output.length; ++i) {
            const currentValue = output[i];
            totalValue += currentValue;
            probabilityArray.push(currentValue);
            if (currentValue > highA) {
                guess = i;
                highA = currentValue;
            }
        }
        const normaliseProbabilityArray = probabilityArray.map(item => (item / totalValue) * 100);
        return [guess, normaliseProbabilityArray];
    };
}
