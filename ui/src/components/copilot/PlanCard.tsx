"use client";

import { Button } from "@/components/ui/button";
import type { Plan } from "@/lib/copilot/types";

interface PlanCardProps {
    plan: Plan;
    /** false once the user answered (or a turn is running): buttons disable */
    actionable: boolean;
    onApprove: () => void;
    onChangeSomething: () => void;
}

export function PlanCard({ plan, actionable, onApprove, onChangeSomething }: PlanCardProps) {
    return (
        <section aria-label="Proposed plan" className="space-y-3 rounded-lg border bg-background p-3">
            <div className="space-y-1">
                <h3 className="text-sm font-medium">Plan</h3>
                <p className="whitespace-pre-wrap break-words text-sm">{plan.summary}</p>
            </div>
            {plan.steps.length > 0 ? (
                <ol className="list-decimal space-y-2 pl-5 text-sm tabular-nums">
                    {plan.steps.map((step, i) => (
                        <li key={i} className="break-words">
                            <span>{step.title}</span>
                            {step.detail ? (
                                <span className="block text-muted-foreground">{step.detail}</span>
                            ) : null}
                        </li>
                    ))}
                </ol>
            ) : null}
            {plan.questions.length > 0 ? (
                <div className="space-y-1">
                    <h4 className="text-xs font-medium text-muted-foreground">Open questions</h4>
                    <ul className="list-disc space-y-1 pl-5 text-sm">
                        {plan.questions.map((q, i) => (
                            <li key={i} className="break-words">
                                {q}
                            </li>
                        ))}
                    </ul>
                </div>
            ) : null}
            {actionable ? (
                <div className="flex flex-wrap gap-2">
                    <Button className="min-h-11 md:min-h-9" onClick={onApprove}>
                        Approve plan
                    </Button>
                    <Button variant="outline" className="min-h-11 md:min-h-9" onClick={onChangeSomething}>
                        Change something
                    </Button>
                </div>
            ) : null}
        </section>
    );
}
