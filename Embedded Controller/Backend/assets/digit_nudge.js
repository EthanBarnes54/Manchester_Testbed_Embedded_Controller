(function () {
  "use strict";

  function isDigit(ch) {
    return ch >= "0" && ch <= "9";
  }

  function parseLimit(el, attr) {
    var val = el.getAttribute(attr);
    if (val === null || val === "") {
      return null;
    }
    var num = parseFloat(val);
    return Number.isFinite(num) ? num : null;
  }

  function resolveDigitIndex(value, caret) {
    if (caret === null || caret === undefined) {
      return null;
    }
    if (caret < value.length && isDigit(value[caret])) {
      return caret;
    }
    if (caret > 0 && isDigit(value[caret - 1])) {
      return caret - 1;
    }
    return null;
  }

  function applyDeltaWithCarry(value, index, delta) {
    var chars = value.split("");
    var i = index;
    var carry = delta;

    while (i >= 0 && carry !== 0) {
      if (!isDigit(chars[i])) {
        i -= 1;
        continue;
      }
      var digit = parseInt(chars[i], 10) + carry;
      if (digit >= 0 && digit <= 9) {
        chars[i] = String(digit);
        carry = 0;
        break;
      }
      if (digit > 9) {
        chars[i] = "0";
        carry = 1;
        i -= 1;
        continue;
      }
      chars[i] = "9";
      carry = -1;
      i -= 1;
    }

    if (carry === 1) {
      chars.unshift("1");
      return { value: chars.join(""), caretShift: 1 };
    }
    if (carry === -1) {
      return null;
    }
    return { value: chars.join(""), caretShift: 0 };
  }

  function handleKeydown(e) {
    if (e.key !== "ArrowUp" && e.key !== "ArrowDown") {
      return;
    }
    var el = e.target;
    if (!el || !el.classList || !el.classList.contains("digit-nudge")) {
      return;
    }
    if (el.selectionStart !== el.selectionEnd) {
      return;
    }

    var raw = el.value || "";
    var caret = el.selectionStart;
    if (!raw) {
      raw = "0";
      caret = 0;
    }

    var digitIndex = resolveDigitIndex(raw, caret);
    if (digitIndex === null) {
      return;
    }

    var delta = e.key === "ArrowUp" ? 1 : -1;
    var result = applyDeltaWithCarry(raw, digitIndex, delta);
    if (!result) {
      return;
    }

    var candidate = result.value;
    var parsed = parseFloat(candidate);
    if (!Number.isFinite(parsed)) {
      return;
    }

    var min = parseLimit(el, "min");
    var max = parseLimit(el, "max");
    if (min !== null && parsed < min) {
      return;
    }
    if (max !== null && parsed > max) {
      return;
    }

    e.preventDefault();
    el.value = candidate;
    el.dispatchEvent(new Event("input", { bubbles: true }));

    var newCaret = Math.min(caret + result.caretShift, el.value.length);
    requestAnimationFrame(function () {
      try {
        el.setSelectionRange(newCaret, newCaret);
      } catch (err) {
        // Ignore browsers that block selection range changes.
      }
    });
  }

  function bind(el) {
    if (el.dataset.digitNudgeBound === "1") {
      return;
    }
    el.addEventListener("keydown", handleKeydown);
    el.dataset.digitNudgeBound = "1";
  }

  function scan() {
    var inputs = document.querySelectorAll("input.digit-nudge");
    for (var i = 0; i < inputs.length; i += 1) {
      bind(inputs[i]);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", scan);
  } else {
    scan();
  }

  var observer = new MutationObserver(scan);
  observer.observe(document.documentElement, { childList: true, subtree: true });
})();
