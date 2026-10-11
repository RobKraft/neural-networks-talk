// 7-segment perceptron training for digits 1, 3, 8
// Run with: node console/train-1-3-8.js
//
// Watch digit 3: the network gets it right at step 3, but the fix for 8
// at step 4 breaks it again. A correct answer doesn't mean the weights are done.

// Segment order: [a b c d e f g]
const digits = {
  1: [0, 1, 1, 0, 0, 0, 0],
  3: [1, 1, 1, 1, 0, 0, 1],
  8: [1, 1, 1, 1, 1, 1, 1]
};

// Training order
const trainingSequence = [3, 1, 3, 8, 3];

// Initialize weights to zero
let weights = {
  1: Array(7).fill(0),
  3: Array(7).fill(0),
  8: Array(7).fill(0)
};

// Utility: dot product
function dot(a, b) {
  return a.reduce((sum, val, i) => sum + val * b[i], 0);
}

// Predict the digit with the highest score (ties go to the lowest digit)
function predict(input) {
  let best = null;
  let bestScore = -Infinity;
  for (const d of [1, 3, 8]) {
    const score = dot(weights[d], input);
    if (score > bestScore) {
      bestScore = score;
      best = d;
    }
  }
  return best;
}

function report() {
  return [1, 3, 8].map(d => `${d}${predict(digits[d]) === d ? "✓" : "✗"}`).join(" ");
}

trainingSequence.forEach((trainDigit, step) => {
  const x = digits[trainDigit];
  const scores = [1, 3, 8].map(d => `${d}=${dot(weights[d], x)}`).join(", ");
  const predicted = predict(x);
  const updateNeeded = predicted !== trainDigit;

  console.log(`Step ${step + 1}: train on ${trainDigit}`);
  console.log(`  Scores: ${scores}`);
  console.log(`  Predicted: ${predicted} ${updateNeeded ? "(wrong, updating)" : "(correct)"}`);

  if (updateNeeded) {
    // Correct digit += x, wrong prediction -= x
    weights[trainDigit] = weights[trainDigit].map((w, i) => w + x[i]);
    weights[predicted] = weights[predicted].map((w, i) => w - x[i]);
  }

  console.log(`  Weights after step:`);
  console.log(`    w1: [${weights[1].join(", ")}]`);
  console.log(`    w3: [${weights[3].join(", ")}]`);
  console.log(`    w8: [${weights[8].join(", ")}]`);
  console.log(`  Now correct: ${report()}`);
  console.log("------------------------------------------------");
});
