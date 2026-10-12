# Neural Networks, from the ground up

Example code for Rob Kraft's Kansas City user group talk on neural networks.

## ▶ Try it live: **https://robkraft.github.io/neural-networks-talk/**

The **Perceptron Playground** shows a tiny neural network learning to read a
7-segment display (the kind on an alarm clock). There's nothing to install. Open
the link, press **Step**, and watch it:

1. look at a digit's lit segments (7 inputs, each 0 or 1),
2. score every possible answer (`weights · input`),
3. pick the highest score, and
4. if it guessed wrong, **add** the input to the correct digit's weights and
   **subtract** it from the wrong guess's weights.

That's the perceptron learning rule from 1958. A modern network does essentially
the same thing (guess, measure how wrong, adjust weights) billions of times, with
billions of weights.

### Things to try

- **Digits "1, 3, 8" with fixed order.** At step 3 the network gets 3 right.
  At step 4 it fixes its mistake on 8, and because every segment of 3 is also
  in 8, that fix breaks 3. Step 5 repairs it. A set of weights that gives the
  right answer for one input may still need to change. The page flags every
  update that breaks a digit that was already correct.
- **Shuffled order + Reset a few times.** The number of steps and the final
  weights change every run. Many different sets of weights give the same correct
  answers, so training isn't deterministic.
- **Random starting weights.** It still converges, but on yet another set of
  weights.
- **Set "Weight updates" to manual.** Step first shows the next digit with a
  "?" so the audience can guess; Step again reveals the network's guess. Each
  wrong guess pauses with the cells
  about to change marked +1 / −1, so you can discuss the mistake before
  clicking **Adjust Weights** to apply it.
- **Light up a shape that isn't a digit.** The network still answers confidently,
  because it can only pick from what it was trained on.
- **Pick any weight and explain it.** You can't, which is why nobody can just
  "fix the bug" when a model gives a wrong answer.

Keyboard shortcuts: `S` or `→` steps, `Space` plays/pauses, `R` resets, `A` adjusts weights (manual mode).

## Run the console versions

The original scripts from the talk are in [`console/`](console). They need
[Node.js](https://nodejs.org/) and nothing else:

```bash
node console/train-1-3-8.js        # 3 is learned, then fixing 8 breaks it
node console/train-3-8-9.js        # warm-up: learn just 3, 8 and 9
node console/train-all-digits.js   # learn all ten digits until every one is correct
```

## Run the web page locally

It's plain HTML, CSS and JavaScript with no build step. Either open `index.html`
directly in a browser, or serve the folder:

```bash
npx http-server . -p 8080
```

## Files

| File | What it is |
| --- | --- |
| `index.html`, `styles.css`, `app.js` | The interactive Perceptron Playground |
| `console/train-1-3-8.js` | Console demo: a correct answer for 3 gets broken by the fix for 8 |
| `console/train-3-8-9.js` | Console demo: three digits, fixed training sequence |
| `console/train-all-digits.js` | Console demo: all ten digits, trains until 10/10 |

## License

[MIT](LICENSE). Use it in your own talks and classes.
