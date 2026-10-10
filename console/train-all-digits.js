// 7-segment perceptron training for digits 0-9
// Run with: node console/train-all-digits.js

// Segment order: [a b c d e f g]
const digits = {
  0: [1, 1, 1, 1, 1, 1, 0],
  1: [0, 1, 1, 0, 0, 0, 0],
  2: [1, 1, 0, 1, 1, 0, 1],
  3: [1, 1, 1, 1, 0, 0, 1],
  4: [0, 1, 1, 0, 0, 1, 1],
  5: [1, 0, 1, 1, 0, 1, 1],
  6: [1, 0, 1, 1, 1, 1, 1],
  7: [1, 1, 1, 0, 0, 0, 0],
  8: [1, 1, 1, 1, 1, 1, 1],
  9: [1, 1, 1, 1, 0, 1, 1]
};

// Initialize weights to zero for all digits
let weights = {};
for (let d = 0; d <= 9; d++) {
  weights[d] = Array(7).fill(0);
}

// Utility: dot product
function dot(a, b) {
  return a.reduce((sum, val, i) => sum + val * b[i], 0);
}

// Predict the digit with the highest score
function predict(input) {
  let bestDigit = 0;
  let bestScore = -Infinity;
  for (let d = 0; d <= 9; d++) {
    const score = dot(weights[d], input);
    if (score > bestScore || (score === bestScore && d < bestDigit)) {
      bestScore = score;
      bestDigit = d;
    }
  }
  return bestDigit;
}

// Check if all digits are correctly classified
function allCorrect() {
  for (let d = 0; d <= 9; d++) {
    if (predict(digits[d]) !== d) return false;
  }
  return true;
}

let step = 0;
let epoch = 0;
const MAX_STEPS = 100000; // Safety cap

while (!allCorrect() && step < MAX_STEPS) {
  epoch++;
  for (let trainDigit = 0; trainDigit <= 9; trainDigit++) {
    const predicted = predict(digits[trainDigit]);
    const updateNeeded = predicted !== trainDigit;

    const scores = {};
    for (let d = 0; d <= 9; d++) {
      scores[d] = dot(weights[d], digits[trainDigit]);
    }

    step++;
    console.log(`Step ${step} (Epoch ${epoch}): train on ${trainDigit}`);
    console.log(`  Scores: ${Object.entries(scores).map(([d, s]) => `${d}=${s}`).join(", ")}`);
    console.log(`  Predicted: ${predicted} ${updateNeeded ? "(wrong, updating)" : "(correct)"}`);

    if (updateNeeded) {
      const x = digits[trainDigit];
      weights[trainDigit] = weights[trainDigit].map((w, i) => w + x[i]);
      weights[predicted]  = weights[predicted].map((w, i) => w - x[i]);
    }

    console.log(`  Weights after step:`);
    for (let d = 0; d <= 9; d++) {
      console.log(`    w${d}: [${weights[d].join(", ")}]`);
    }
    console.log("------------------------------------------------");

    if (allCorrect()) break; // Stop mid-epoch if already converged
  }
}

// Final report
console.log("\n=== Training Complete ===");
console.log(`Converged in ${step} steps across ${epoch} epochs\n`);
console.log("=== Final Accuracy Check ===");
let correct = 0;
for (let d = 0; d <= 9; d++) {
  const predicted = predict(digits[d]);
  if (predicted === d) correct++;
  console.log(`Digit ${d}: predicted=${predicted} ${predicted === d ? "✓" : "✗"}`);
}
console.log(`\nAccuracy: ${correct}/10`);
