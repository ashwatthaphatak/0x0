// src/components/protection-level-selector.tsx
"use client";

import { PROTECTION_LEVELS } from "@/types";
import type { ProtectionLevel } from "@/types";

interface Props {
  level:    ProtectionLevel;
  onChange: (level: ProtectionLevel) => void;
  disabled?: boolean;
}

export function ProtectionLevelSelector({ level, onChange, disabled }: Props) {
  return (
    <div className="flex flex-col gap-2">
      <span className="text-xs font-semibold uppercase tracking-widest text-slate-400">
        Protection Level
      </span>
      <div className="flex gap-2">
        {(Object.keys(PROTECTION_LEVELS) as ProtectionLevel[]).map((l) => {
          const cfg    = PROTECTION_LEVELS[l];
          const active = level === l;
          return (
            <button
              key={l}
              disabled={disabled}
              onClick={() => onChange(l)}
              className={[
                "flex-1 flex flex-col items-center px-3 py-2.5 rounded-xl border",
                "transition-all duration-200 text-sm",
                active
                  ? "border-indigo-500 bg-indigo-950/60 text-white"
                  : "border-slate-700 bg-slate-800/40 text-slate-400 hover:border-slate-500 hover:text-slate-200",
                disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer",
              ].join(" ")}
            >
              <span className="font-semibold">{cfg.label}</span>
              <span className="text-[10px] opacity-70 mt-0.5">{cfg.description}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
