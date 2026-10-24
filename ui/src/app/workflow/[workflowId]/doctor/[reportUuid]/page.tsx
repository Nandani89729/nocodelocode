"use client";

import { useParams } from "next/navigation";

import { DoctorReportView } from "@/components/doctor/DoctorReportView";

import WorkflowLayout from "../../../WorkflowLayout";

export default function DoctorReportPage() {
    const { workflowId, reportUuid } = useParams<{ workflowId: string; reportUuid: string }>();

    return (
        <WorkflowLayout showFeaturesNav={false}>
            <DoctorReportView workflowId={Number(workflowId)} reportUuid={reportUuid} />
        </WorkflowLayout>
    );
}
