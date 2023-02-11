import * as math from 'mathjs';
import { Matrix } from 'mathjs';

export const randomNumber = (min: number, max: number): number => {
    return Math.random() * (max - min) + min;
};

export const randomMatrix = (row: number, column: number): Matrix => {
    const m = math.map(math.ones(row, column), elm => {
        return elm * randomNumber(-2, 2);
    }) as Matrix;
    return m;
};

export const zeroMatrix = (row: number, column: number): Matrix => {
    const m = math.zeros(row, column, 'dense') as Matrix;
    return m;
};

export const onesMatrix = (row: number, column: number): Matrix => {
    const m = math.ones(row, column, 'dense') as Matrix;
    return m;
};
