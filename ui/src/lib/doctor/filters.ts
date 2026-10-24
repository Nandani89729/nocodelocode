import { formatDateRange, formatNumberRange } from "@/lib/filters";
import type {
    ActiveFilter,
    DateRangeValue,
    MultiSelectValue,
    NumberRangeValue,
    NumberValue,
    RadioValue,
    TextValue,
} from "@/types/filters";

import type { DoctorRunFilter } from "./types";

/** Same shape WorkflowExecutions sends to the runs list endpoint. */
export const toDoctorFilters = (filters: ActiveFilter[]): DoctorRunFilter[] =>
    filters.map((f) => ({ attribute: f.attribute.id, type: f.attribute.type, value: f.value }));

function valueInWords(filter: ActiveFilter): string {
    const { attribute, value } = filter;
    switch (attribute.type) {
        case "dateRange":
            return formatDateRange(value as DateRangeValue);
        case "multiSelect":
        case "tags": {
            const codes = (value as MultiSelectValue).codes ?? [];
            return codes.length ? codes.join(", ") : "any";
        }
        case "numberRange":
            return formatNumberRange(value as NumberRangeValue, attribute.config.unit);
        case "number":
        case "numberSelect": {
            const v = (value as NumberValue).value;
            const option = attribute.config.numberSelectOptions?.find((o) => o.value === v);
            return option?.label ?? (v === null || v === undefined ? "any" : String(v));
        }
        case "radio": {
            const status = (value as RadioValue).status;
            return attribute.config.radioOptions?.find((o) => o.value === status)?.label ?? status;
        }
        case "text":
            return `"${(value as TextValue).value}"`;
        default:
            return "";
    }
}

/** "Status: Completed; Duration: 10 - 60 seconds" or "No filters". */
export function filtersInWords(filters: ActiveFilter[]): string {
    if (filters.length === 0) return "No filters: latest completed calls";
    return filters.map((f) => `${f.attribute.label}: ${valueInWords(f)}`).join("; ");
}
