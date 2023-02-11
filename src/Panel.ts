import { Point, LINE_WIDTH, SCREEN_WIDTH, SCREEN_HEIGHT, INPUT_SIZE } from './constants';
import { MnistEntry } from './MnistEntry';

export class Panel {
    private path: Point[];
    private clicked: boolean;
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D;

    constructor(canvas: HTMLCanvasElement) {
        this.path = [];
        this.clicked = false;
        this.canvas = canvas;
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        this.ctx = canvas.getContext('2d')!;

        this.canvas.addEventListener('mouseup', () => {
            this.clicked = false;
            this.path = [];
        });

        this.canvas.addEventListener('mousedown', (e: MouseEvent) => {
            this.clicked = true;
            this.drawPoint(this.getPos(e));
        });

        this.canvas.addEventListener('mousemove', (e: MouseEvent) => {
            if (this.clicked) this.drawLine(this.getPos(e));
        });
    }

    private getPos = (e: MouseEvent) => {
        e.preventDefault();

        const rect = this.canvas.getBoundingClientRect();

        const _x = this.canvas.width / rect.width;
        const _y = this.canvas.height / rect.height;

        const x = e.offsetX * _x;
        const y = e.offsetY * _y;
        return { x, y };
    };

    private drawLine = (p: Point) => {
        this.path.push({ x: p.x, y: p.y });

        if (this.path.length > 1) {
            this.ctx.beginPath();
            this.ctx.lineWidth = LINE_WIDTH;
            this.ctx.lineCap = 'round';
            this.ctx.moveTo(this.path[this.path.length - 2].x, this.path[this.path.length - 2].y);
            this.ctx.lineTo(this.path[this.path.length - 1].x, this.path[this.path.length - 1].y);
            this.ctx.stroke();
            this.path.shift();
        }
    };

    private drawPoint = (p: Point) => {
        if (!this.path.length) {
            this.path.push({ x: p.x, y: p.y });
            this.ctx.beginPath();
            // this.ctx.arc(p.x, p.y, LINE_WIDTH / 2, 0, Math.PI * 2);
            this.ctx.arc(p.x, p.y, 1, 0, Math.PI * 2);
            this.ctx.fill();
        }
    };

    erase = () => {
        this.ctx.save();
        this.ctx.clearRect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT);
        this.ctx.fillStyle = 'white';
        this.ctx.fillRect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT);
        this.ctx.restore();
    };

    drawMnist = (mnist: MnistEntry): void => {
        const smallBoxSize = 7;
        this.ctx.save();
        const dataArray = mnist.imageArray;
        for (let i = 0; i < INPUT_SIZE; ++i) {
            const colorValue = dataArray[i];
            if (colorValue > 0) {
                const invertedColur = Math.floor(255 - colorValue * 255); // because previously we divide 255 for normalisation
                console.log(`rgb(${invertedColur}, ${invertedColur}, ${invertedColur})`);
                this.ctx.fillStyle = `rgb(${invertedColur}, ${invertedColur}, ${invertedColur})`;
                this.ctx.fillRect(
                    (i % 28) * smallBoxSize,
                    Math.floor(i / 28) * smallBoxSize,
                    smallBoxSize,
                    smallBoxSize
                );
            }
        }
        this.ctx.restore();
    };
}
