<?php

namespace app\Http\Controllers\RecycleController;
use App\Http\Controllers\HomeController;
use app\Http\Controllers\RecycleController;
use Illuminate\Support\Facades\Route;

Route::get('/', [HomeController::class, 'index']);
Route::post('/classify', [RecycleController::class, 'classify']);
