/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { ChangePassword } from '../models/ChangePassword';
import type { UpdateUser } from '../models/UpdateUser';
import type { User } from '../models/User';
import type { UserResponse } from '../models/UserResponse';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class ProfileService {
    /**
     * @returns User
     * @throws ApiError
     */
    public static profileRetrieve(): CancelablePromise<User> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/profile/',
        });
    }
    /**
     * @param requestBody
     * @returns UserResponse
     * @throws ApiError
     */
    public static profileChangePasswordCreate(
        requestBody: ChangePassword,
    ): CancelablePromise<UserResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/profile/change-password/',
            body: requestBody,
            mediaType: 'application/json',
        });
    }
    /**
     * @param requestBody
     * @returns UserResponse
     * @throws ApiError
     */
    public static profileUpdateUserUpdate(
        requestBody: UpdateUser,
    ): CancelablePromise<UserResponse> {
        return __request(OpenAPI, {
            method: 'PUT',
            url: '/api/profile/update-user/',
            body: requestBody,
            mediaType: 'application/json',
        });
    }
}
