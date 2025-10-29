package demo.generics;

import java.util.List;

public class Box<T> {
    private final T value;

    public Box(T value) {
        this.value = value;
    }

    public T get() {
        return value;
    }

    public static <U extends Number> U first(List<U> items) {
        return items.get(0);
    }
}
