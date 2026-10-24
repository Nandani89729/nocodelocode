import * as React from "react"

import { cn } from "@/lib/utils"

interface ChoiceChipsProps {
    options: {
        value: string;
        label: string;
    }[];
    value: string;
    onChange: (value: string) => void;
    className?: string;
    /** Extra classes for each chip (e.g. touch-target height). */
    chipClassName?: string;
    /** Names the group for assistive tech. */
    "aria-label"?: string;
    "aria-labelledby"?: string;
    disabled?: boolean;
}

export function ChoiceChips({
    options,
    value,
    onChange,
    className,
    chipClassName,
    disabled,
    ...aria
}: ChoiceChipsProps) {
    return (
        <div role="group" className={cn("flex gap-2 p-4", className)} {...aria}>
            {options.map((option) => (
                <button
                    key={option.value}
                    type="button"
                    aria-pressed={value === option.value}
                    disabled={disabled}
                    onClick={() => onChange(option.value)}
                    className={cn(
                        "px-4 py-2 rounded-full text-sm font-medium transition-all disabled:opacity-50",
                        value === option.value
                            ? "bg-primary text-primary-foreground"
                            : "bg-secondary text-secondary-foreground hover:bg-secondary/80",
                        chipClassName
                    )}
                >
                    {option.label}
                </button>
            ))}
        </div>
    )
}
