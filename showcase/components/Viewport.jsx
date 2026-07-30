"use client";

// Error boundary for the WebGL players.
//
// react-three-fiber throws during render when a WebGL context cannot be created
// — no hardware acceleration, a blocked context, a machine with GPU sandboxing.
// React unwinds to the nearest boundary, and without one that is the root: the
// whole page unmounts and the visitor gets "Application error" instead of the
// paper. Every 3D mount on this site is wrapped in one of these, so a missing
// context costs one figure and nothing else.

import { Component } from "react";

export default class Viewport extends Component {
  constructor(props) {
    super(props);
    this.state = { failed: false };
  }

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(err) {
    if (typeof console !== "undefined") console.warn("3D viewport unavailable:", err?.message || err);
  }

  render() {
    const { children, height = 410, label = "3D playback" } = this.props;
    if (!this.state.failed) return children;
    return (
      <div
        className="grid place-items-center rounded-xl border border-dashed border-[var(--hairline-strong)] bg-black/20 px-6 text-center"
        style={{ height }}
      >
        <div>
          <div className="label mb-2">{label} unavailable</div>
          <p className="mx-auto max-w-xs text-[12px] leading-relaxed text-[var(--muted)]">
            This browser could not open a WebGL context, so the skeleton cannot be animated. The
            tables and charts on this page are unaffected.
          </p>
        </div>
      </div>
    );
  }
}
