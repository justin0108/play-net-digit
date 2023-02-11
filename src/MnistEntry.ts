import { DigitData } from './constants';

export class MnistEntry {
    imageArray: DigitData;
    imageLabel: number;

    constructor(imageArray: DigitData, imageLabel: number) {
        this.imageArray = imageArray;
        this.imageLabel = imageLabel;
    }
}
