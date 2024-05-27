/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export { ApiError } from './core/ApiError';
export { CancelablePromise, CancelError } from './core/CancelablePromise';
export { OpenAPI } from './core/OpenAPI';
export type { OpenAPIConfig } from './core/OpenAPI';

export type { DemoDocs as DemoDocsDto } from './models/DemoDocs';
export type { Metric as MetricDto } from './models/Metric';
export type { RecognizeImage as RecognizeImageDto } from './models/RecognizeImage';

export { DemodocsService } from './services/DemodocsService';
export { MetricsService } from './services/MetricsService';
export { RecognizeService } from './services/RecognizeService';
export { SchemaService } from './services/SchemaService';

import { QueryClient } from "@tanstack/react-query";

export const queryClient = new QueryClient();