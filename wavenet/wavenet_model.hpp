#pragma once

#include "arena.hpp"
#include "wavenet_layer_array.hpp"

namespace wavenet
{
template <typename T,
          int condition_size,
          typename... LayerArrays>
struct Wavenet_Model
{
    std::tuple<LayerArrays...> layer_arrays;

    static constexpr auto head_layer_n_channels = std::tuple_element_t<0, std::tuple<LayerArrays...>>::n_channels;

#if RTNEURAL_USE_EIGEN
    Eigen::Matrix<T, head_layer_n_channels, 1> head_input {};
#elif RTNEURAL_USE_XSIMD
    xsimd::batch<T> head_input[RTNeural::ceil_div (head_layer_n_channels, (int) xsimd::batch<T>::size)];
#endif
    T head_scale = (T) 0;
    T extra_condition_ = (T) 0;

    Memory_Arena<> arena {};

    Wavenet_Model() = default;

    void set_condition (T value) { extra_condition_ = value; }

    void prepare (int block_size)
    {
#if RTNEURAL_USE_EIGEN
        size_t arena_bytes_needed = sizeof (Eigen::Matrix<T, head_layer_n_channels, 1>) * block_size
                                    + sizeof (Eigen::Matrix<T, condition_size, 1>) * block_size;
#elif RTNEURAL_USE_XSIMD
        size_t arena_bytes_needed = sizeof (xsimd::batch<T>[RTNeural::ceil_div (head_layer_n_channels, (int) xsimd::batch<T>::size)]) * block_size
                                    + sizeof (xsimd::batch<T>) * block_size
                                    + sizeof (xsimd::batch<T>[RTNeural::ceil_div (condition_size, (int) xsimd::batch<T>::size)]) * block_size;
#endif
        RTNeural::modelt_detail::forEachInTuple (
            [&arena_bytes_needed, block_size] (auto& layer_array, auto)
            {
                arena_bytes_needed += layer_array.get_arena_bytes_needed (block_size);
            },
            layer_arrays);

        arena.resize (arena_bytes_needed + 256);
        // prewarm();
    }

    void prewarm()
    {
        RTNeural::modelt_detail::forEachInTuple (
            [] (auto& layer, size_t)
            {
                layer.reset();
            },
            layer_arrays);
        for (int i = 0; i < 1 << 14; ++i)
            forward (0.0f);
    }

    void load_weights (const nlohmann::json& model_json)
    {
        std::vector<float> model_weights = model_json.at ("weights");
        auto weights_iterator = model_weights.begin();
        RTNeural::modelt_detail::forEachInTuple (
            [&weights_iterator] (auto& layer, size_t)
            {
                layer.load_weights (weights_iterator);
            },
            layer_arrays);

        head_scale = *weights_iterator++;

        // Make sure we use the all of the weights exactly
        assert (std::distance (model_weights.begin(), weights_iterator) == model_weights.size());
    }

    T forward (T input) noexcept
    {
#if RTNEURAL_USE_EIGEN
        const auto v_ins = Eigen::Matrix<T, 1, 1>::Constant (input);

        // Build condition vector: for condition_size=1 it's just [input],
        // for condition_size=2 it's [input, extra_condition_]
        Eigen::Matrix<T, condition_size, 1> v_cond;
        v_cond (0) = input;
        if constexpr (condition_size >= 2)
            v_cond (1) = extra_condition_;
#elif RTNEURAL_USE_XSIMD
        xsimd::batch<T> v_ins[1];
        v_ins[0] = RTNeural::set_value (v_ins[0], 0, input);

        xsimd::batch<T> v_cond[RTNeural::ceil_div (condition_size, (int) xsimd::batch<T>::size)];
        v_cond[0] = RTNeural::set_value (v_cond[0], 0, input);
        if constexpr (condition_size >= 2)
            v_cond[0] = RTNeural::set_value (v_cond[0], 1, extra_condition_);
#endif

        RTNeural::modelt_detail::forEachInTuple (
            [this, v_ins, v_cond] (auto& layer_array, auto index_t)
            {
                static constexpr size_t index = index_t;
                if constexpr (index == 0)
                {
#if RTNEURAL_USE_EIGEN
                    head_input.setZero();
                    std::get<0> (layer_arrays).forward (v_ins, v_cond, head_input);
#elif RTNEURAL_USE_XSIMD
                    std::fill (std::begin (head_input), std::end (head_input), xsimd::batch<T> {});
                    std::get<0> (layer_arrays).forward (v_ins, v_cond, head_input);
#endif
                }
                else
                {
                    std::get<index> (layer_arrays).forward (std::get<index - 1> (layer_arrays).layer_outputs, v_cond, std::get<index - 1> (layer_arrays).head_outputs);
                }
            },
            layer_arrays);

#if RTNEURAL_USE_EIGEN
        return std::get<std::tuple_size_v<decltype (layer_arrays)> - 1> (layer_arrays).head_outputs[0] * head_scale;
#elif RTNEURAL_USE_XSIMD
        return std::get<std::tuple_size_v<decltype (layer_arrays)> - 1> (layer_arrays).head_outputs[0].get (0) * head_scale;
#endif
    }

    void forward (const T* input, T* output, int N) noexcept
    {
#if RTNEURAL_USE_EIGEN
        // Build per-sample condition vectors
        auto* v_cond = arena.template allocate<Eigen::Matrix<T, condition_size, 1>> (N, RTNEURAL_DEFAULT_ALIGNMENT);
        for (int n = 0; n < N; ++n)
        {
            v_cond[n] (0) = input[n];
            if constexpr (condition_size >= 2)
                v_cond[n] (1) = extra_condition_;
        }
        const auto* v_ins = reinterpret_cast<const Eigen::Matrix<T, 1, 1>*> (input);
#elif RTNEURAL_USE_XSIMD
        auto* v_ins = arena.allocate<xsimd::batch<T>[1]> (N, RTNEURAL_DEFAULT_ALIGNMENT);
        for (int n = 0; n < N; ++n)
            v_ins[n][0] = RTNeural::set_value (v_ins[n][0], 0, input[n]);

        auto* v_cond = arena.allocate<xsimd::batch<T>[RTNeural::ceil_div (condition_size, (int) xsimd::batch<T>::size)]> (N, RTNEURAL_DEFAULT_ALIGNMENT);
        for (int n = 0; n < N; ++n)
        {
            v_cond[n][0] = RTNeural::set_value (v_cond[n][0], 0, input[n]);
            if constexpr (condition_size >= 2)
                v_cond[n][0] = RTNeural::set_value (v_cond[n][0], 1, extra_condition_);
        }
#endif

        RTNeural::modelt_detail::forEachInTuple (
            [this, v_ins, v_cond, N, output] (auto& layer_array, auto index_t)
            {
                static constexpr size_t index = index_t;
                if constexpr (index == 0)
                {
#if RTNEURAL_USE_EIGEN
                    auto* head_inputs = arena.allocate<Eigen::Matrix<T, head_layer_n_channels, 1>> (N, RTNEURAL_DEFAULT_ALIGNMENT);
                    for (int n = 0; n < N; ++n)
                        head_inputs[n].setZero();
                    std::get<0> (layer_arrays).forward (v_ins, v_cond, head_inputs, N, arena);
#elif RTNEURAL_USE_XSIMD
                    auto* head_inputs = arena.allocate<xsimd::batch<T>[RTNeural::ceil_div (head_layer_n_channels, (int) xsimd::batch<T>::size)]> (N, RTNEURAL_DEFAULT_ALIGNMENT);
                    for (int n = 0; n < N; ++n)
                        std::fill (std::begin (head_inputs[n]), std::end (head_inputs[n]), xsimd::batch<T> {});
                    std::get<0> (layer_arrays).forward (v_ins, v_cond, head_inputs, N, arena);
#endif
                }
                else
                {
                    auto& prev_layer_array = std::get<index - 1> (layer_arrays);
                    std::get<index> (layer_arrays).forward (prev_layer_array.layer_outputs_arr, v_cond, prev_layer_array.head_outputs_arr, N, arena);
                }
            },
            layer_arrays);

        auto& last_layer_array = std::get<std::tuple_size_v<decltype (layer_arrays)> - 1> (layer_arrays);
        for (int n = 0; n < N; ++n)
        {
#if RTNEURAL_USE_EIGEN
            output[n] = last_layer_array.head_outputs_arr[n][0] * head_scale;
#elif RTNEURAL_USE_XSIMD
            output[n] = last_layer_array.head_outputs_arr[n][0].get (0) * head_scale;
#endif
        }

        arena.clear();
    }
};
} // namespace wavenet
