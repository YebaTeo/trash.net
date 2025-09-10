<?php

namespace app\Http\Controllers;
use App\Http\Controllers\Controller;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;

class RecycleController extends Controller
{
    public function classify(Request $request)
    {
        $response = Http::post('http://127.0.0.1:5000/classify', [
            'img_url' => $request->input('img_url')
        ]);

        if ($response->successful()) {
            return response()->json($response->json());
        } else {
            return response()->json(['error' => 'Gagal menghubungi Python API'], 500);
        }
    }
}
