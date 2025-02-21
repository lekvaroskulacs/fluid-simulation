import { Renderer } from "./renderer";

const canvas: HTMLCanvasElement = <HTMLCanvasElement>document.getElementById("gfx-main");
canvas.setAttribute("width", window.innerWidth.toString());
canvas.setAttribute("height", window.innerHeight.toString());

const renderer = new Renderer(canvas);

renderer.init();

window.addEventListener("resize", () => {
    const form = <HTMLElement> document.getElementById("form");
    const formHeight = form.offsetHeight;
    canvas.setAttribute("width", window.innerWidth.toString());
    canvas.setAttribute("height", (window.innerHeight - formHeight).toString());
    renderer.render();
});


//renderParticles(new Circle(100, 100, 50));


