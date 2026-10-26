// Покадровый рендер src/video.html.
//   node tools/render.js preview 0.5 2 4 ...   -> build/preview/t_XX.jpg (отдельные кадры)
//   node tools/render.js video                 -> build/frames.mp4 (без звука, 30 fps)
const { chromium } = require('playwright');
const { spawn } = require('child_process');
const fs = require('fs'), path = require('path');

const ROOT = path.resolve(__dirname, '..');
const FPS = 30;
(async () => {
  const [mode, ...rest] = process.argv.slice(2);
  fs.mkdirSync(path.join(ROOT, 'build/preview'), { recursive: true });
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1080, height: 1920 } });
  await page.goto('file://' + path.join(ROOT, 'src/video.html'));
  await page.evaluate(() => document.fonts.ready);
  await page.evaluate(() => Promise.all([...document.images].map(i => i.decode())));

  if (mode === 'preview') {
    for (const t of rest.map(Number)) {
      await page.evaluate(t => render(t), t);
      await page.screenshot({ path: path.join(ROOT, `build/preview/t_${t.toFixed(2)}.jpg`), type: 'jpeg', quality: 80 });
    }
  } else {
    const dur = await page.evaluate(() => DURATION);
    const n = Math.round(dur * FPS);
    const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(FPS), '-i', '-',
      '-c:v', 'libx264', '-preset', 'slow', '-crf', '16', '-pix_fmt', 'yuv420p', path.join(ROOT, 'build/frames.mp4')], { stdio: ['pipe', 'inherit', 'inherit'] });
    for (let i = 0; i < n; i++) {
      await page.evaluate(t => render(t), i / FPS);
      const buf = await page.screenshot({ type: 'jpeg', quality: 95 });
      if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
      if (i % 60 === 0) process.stdout.write(`frame ${i}/${n}\n`);
    }
    ff.stdin.end();
    await new Promise(r => ff.on('close', r));
  }
  await browser.close();
})();
