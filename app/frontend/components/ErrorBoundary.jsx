"use client";

// Shows a component crash ON THE PAGE instead of replacing the whole app with
// Next.js's generic "Application error: a client-side exception has occurred".
//
// The point is diagnosis without a console: Safari has no built-in one, so a
// blank error screen leaves nothing to go on. This keeps the rest of the app
// alive, prints the actual message and stack, and offers a reload — which also
// covers the most common cause, a stale bundle left over from a rebuild while
// the tab was open (code-split chunks 404 and React throws on the import).

import { Component } from "react";

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null, info: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    this.setState({ info });
    // Still log it, for anyone who does have a console open.
    console.error("[ErrorBoundary]", error, info);   // eslint-disable-line no-console
  }

  render() {
    const { error, info } = this.state;
    if (!error) return this.props.children;

    const stale = /chunk|dynamically imported module|Importing a module script failed/i
      .test(String(error?.message || ""));

    return (
      <section className="surface p-5">
        <div className="label text-[var(--amber)]">
          {this.props.label ? `${this.props.label} — crashed` : "this panel crashed"}
        </div>
        <p className="mt-2 font-mono text-[13px] text-[var(--amber)]">
          {String(error?.message || error)}
        </p>

        {stale && (
          <p className="mt-3 max-w-2xl text-[12px] leading-relaxed text-slate-300">
            This looks like a <strong>stale bundle</strong>: the page was loaded before the
            frontend was rebuilt, so a code-split chunk it wants no longer exists. A hard
            reload fixes it — Safari: ⌥⌘R, or Develop → Empty Caches.
          </p>
        )}

        <div className="mt-3 flex gap-2">
          <button onClick={() => window.location.reload()} className="btn-signal !px-3 !py-1.5 text-[12px]">
            reload page
          </button>
          <button onClick={() => this.setState({ error: null, info: null })}
                  className="btn-ghost !px-3 !py-1.5 text-[12px]">
            try again
          </button>
        </div>

        {(error?.stack || info?.componentStack) && (
          <details className="mt-3">
            <summary className="cursor-pointer text-[11px] text-[var(--muted)]">
              details (copy this if you want me to look)
            </summary>
            <pre className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap rounded-lg border border-[var(--hairline)] bg-black/40 p-3 text-[10px] leading-relaxed text-[var(--muted)]">
{String(error?.stack || "")}
{info?.componentStack ? `\n--- component stack ---${info.componentStack}` : ""}
            </pre>
          </details>
        )}
      </section>
    );
  }
}
