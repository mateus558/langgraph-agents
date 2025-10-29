package demo.analysis;

import java.util.List;

public class Outer<T extends Number> implements AutoCloseable {
    private int counter = 0;
    protected static final String NAME = "Outer";

    public Outer(int initial) {
        this.counter = initial;
    }

    public int add(int x, int y) throws IllegalArgumentException {
        if (x < 0) {
            throw new IllegalArgumentException("x");
        }
        return x + y + counter;
    }

    @Override
    public void close() {}

    public static class Nested {
        public static double multiply(double a, double b) {
            return a * b;
        }
    }
}

sealed interface Shape permits Circle, Rectangle {}

record Circle(double radius) implements Shape {}

record Rectangle(double width, double height) implements Shape {}
