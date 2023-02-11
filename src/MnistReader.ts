import { DigitData, INPUT_SIZE, LABEL_SIZE, TOTAL_IMAGES } from './constants';
import { MnistEntry } from './MnistEntry';

export class MnistReader {
    load = async (trainingImageFile: File, trainingLabelFile: File): Promise<MnistEntry[]> => {
        const mnistArray: MnistEntry[] = [];
        const imageArray = await this.loadImage(trainingImageFile);
        const labelArray = await this.loadLabel(trainingLabelFile);
        for (let i = 0; i < TOTAL_IMAGES; ++i) {
            mnistArray.push(new MnistEntry(imageArray[i], labelArray[i]));
        }
        console.log(mnistArray.length);
        return mnistArray;
    };

    private loadImage = async (trainingImageFile: File): Promise<DigitData[]> => {
        const imageArray: DigitData[] = [];
        const fileBuffer = await trainingImageFile.arrayBuffer();
        const headerOffset = 16;
        for (let i = 0; i < TOTAL_IMAGES; ++i) {
            const startSliceIndex = headerOffset + i * INPUT_SIZE;
            // normalise all value for better learning
            imageArray.push(
                Array.from(new Uint8Array(fileBuffer.slice(startSliceIndex, startSliceIndex + INPUT_SIZE))).map(
                    item => item / 255
                )
            );
        }
        return imageArray;
    };

    private loadLabel = async (trainingLabelFile: File): Promise<number[]> => {
        const labelArray: number[] = [];
        const fileBuffer = await trainingLabelFile.arrayBuffer();
        const headerOffset = 8;
        for (let i = 0; i < TOTAL_IMAGES; ++i) {
            const startSliceIndex = headerOffset + i * LABEL_SIZE;
            labelArray.push(new Uint8Array(fileBuffer.slice(startSliceIndex, startSliceIndex + INPUT_SIZE))[0]);
        }
        return labelArray;
    };
}
