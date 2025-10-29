package demo.nested;

public class Outer {
    private int value;

    public Outer(int value) {
        this.value = value;
    }

    public class Inner {
        public int square() {
            return value * value;
        }
    }

    public static class StaticNested {
        public static int doubleValue(int x) {
            return x * 2;
        }
    }
}
